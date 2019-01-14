'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
#from torch.nn.modules.loss import _Loss

import torchvision
import torchvision.transforms as transforms

import argparse
import math
import time
from tqdm import tqdm
import numpy as np

from models import GoogLeNet, MobileNetV2, VGG, SENet18
from alrao.custom_layers import LinearClassifier, LinearRegressor
from alrao.optim_spec import SGDAlrao, AdamAlrao
from alrao.learningratesgen import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights
from alrao.switch import log_sum_exp

from alrao.earlystopping import EarlyStopping
from alrao.alrao_model import AlraoModel
# TO BE REMOVED
from alrao.utils import Subset


parser = argparse.ArgumentParser(description='alrao')

# CUDA
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable cuda')

# epochs
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs for phase 1 (default: 50)')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='use early stopping')

# options
parser.add_argument('--model_name', default='GoogLeNet',
                    help='Model {VGG19, GoogLeNet, MobileNetV2, SENet18}')
parser.add_argument('--optimizer', default='SGD',
                    help='optimizer (default: SGD) {Adam, SGD}')
parser.add_argument('--lr', type=float, default=.001,
                    help='learning rate, when used without alrao')
parser.add_argument('--momentum', type=float, default=0.,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='l2 penalty')
parser.add_argument('--data_augm', action='store_true', default=False,
                    help='add data augmentation')
parser.add_argument('--size_multiplier', type=int, default=1,
                    help='multiplier of the number of neurons per layer (default: 1)')

# Alrao Parameters
parser.add_argument('--use_alrao', action='store_true', default=True,
                    help='multiple learning rates')
parser.add_argument('--minLR', type=int, default=-3,  # base = -5
                    help='log10 of the minimum LR in alrao (log_10 eta_min)')
parser.add_argument('--maxLR', type=int, default=-3,  # base = 0
                    help='log10 of the maximum LR in alrao (log_10 eta_max)')
parser.add_argument('--nb_class', type=int, default=1,
                    help='number of classifiers before the switch')
parser.add_argument('--task', default='regression',
                    help='task to perform default: "classification" {"classification", "regression"}')

args = parser.parse_args()

torch.manual_seed(193062818)

use_cuda = True #torch.cuda.is_available()
best_acc = 0  # best test accuracy

batch_size = 32
input_dim = 10
pre_output_dim = 100
data_train_size = 1000
data_test_size = 100
sigma2 = 1.
remove_non_numerical = True
bound = math.pi
func = lambda x: math.sin(x * bound)

def is_numerical(x):
    s = x.sum()
    if (s == float('inf')).item():
        return False
    if (s == float('-inf')).item():
        return False
    if s != s:
        return False
    return True

# Data
def generate_data(f, input_dim, nb_data):
    proto = torch.tensor([0.])
    if use_cuda:
        proto = proto.cuda()

    inputs = proto.new().resize_(nb_data).uniform_(-1., 1.)
    dataset = proto.new().resize_(nb_data, input_dim)
    targets = proto.new().resize_(nb_data, 1)
    for i in range(input_dim):
        dataset[:, i] = inputs.pow(i + 1)

    for i in range(nb_data):
        targets[i, 0] = f(inputs[i])

    return dataset, targets

train_inputs, train_targets = generate_data(func, input_dim, data_train_size)
test_inputs, test_targets = generate_data(func, input_dim, data_test_size)

# Regression loss
# base loss class
class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

# loss adapted to one final layer
class L2LossLog(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', sigma2 = 1.):
        super(L2LossLog, self).__init__(size_average, reduce, reduction)
        self.sigma2 = sigma2

    def forward(self, input, target):
        return ((input - target).pow(2).sum(1) / (2 * self.sigma2)).mean() + \
                .5 * math.log(2 * math.pi * self.sigma2)

# loss adapted to the output of the switch
class L2LossAdditional(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', sigma2 = 1.):
        super(L2LossAdditional, self).__init__(size_average, reduce, reduction)
        self.sigma2 = sigma2

    def forward(self, input, target):
        means, ps = input
        # means: batch_size * out_size * nb_classifiers
        means = means.transpose(2, 1).transpose(1, 0)
        # means: nb_classifiers * batch_size * out_size
        log_probas_per_cl = -(means - target).pow(2).sum(2) / (2 * self.sigma2) - \
                .5 * math.log(2 * math.pi * self.sigma2)
        # log_probas_per_cl: nb_classifiers * batch_size
        log_probas_per_cl = log_probas_per_cl.transpose(0, 1)
        # log_probas_per_cl: batch_size * nb_classifiers
        log_probas_per_cl = log_probas_per_cl + ps.log()
        log_probas = log_sum_exp(log_probas_per_cl, dim = 1)
        return -log_probas.mean()

# Model (pre-classifier)
class RegModel(nn.Module):
    def __init__(self, input_dim, pre_output_dim):
        super(RegModel, self).__init__()
        self.layer = nn.Linear(input_dim, pre_output_dim)
        self.relu = nn.ReLU()
        self.linearinputdim = pre_output_dim

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        return x

def build_preclassifier(model_name, *args, **kwargs):
    return RegModel(*args)

class StandardModel(nn.Module):
    def __init__(self, preclassifier, K=1):
        super(StandardModel, self).__init__()
        self.preclassifier = preclassifier
        self.classifier = LinearClassifier(self.preclassifier.linearinputdim, 10)

    def forward(self, x):
        x = self.preclassifier(x)
        return self.classifier(x)

class StandardModelReg(nn.Module):
    def __init__(self, preclassifier):
        super(StandardModelReg, self).__init__()
        self.preclassifier = preclassifier
        self.regressor = LinearRegressor(self.preclassifier.linearinputdim, 1)

    def forward(self, x):
        x = self.preclassifier(x)
        x = self.regressor(x)
        return x.unsqueeze(2), x.new().resize_(1).fill_(1.)

base_lr = args.lr
minlr = 10 ** args.minLR
maxlr = 10 ** args.maxLR

preclassifier = build_preclassifier(args.model_name, input_dim, pre_output_dim)
criterion = L2LossLog(sigma2 = sigma2)
criterion_add = L2LossAdditional(sigma2 = sigma2)
if args.use_alrao:
    net = AlraoModel(preclassifier, args.nb_class, LinearRegressor,
                     'regression', criterion, preclassifier.linearinputdim, 1)
    total_param = sum(np.prod(p.size()) for p in net.parameters_preclassifier())
    total_param += sum(np.prod(p.size())
                       for lcparams in net.classifiers_parameters_list()
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    if args.task == 'classification':
        net = StandardModel(preclassifier)
    elif args.task == 'regression':
        net = StandardModelReg(preclassifier)
    total_param = sum(np.prod(p.size()) for p in net.parameters())
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))

if use_cuda:
    net.cuda()


if args.use_alrao:
    if args.nb_class > 1:
        classifiers_lr = [np.exp(
            np.log(minlr) + k / (args.nb_class-1) * (np.log(maxlr) - np.log(minlr))
        ) for k in range(args.nb_class)]
        print(("Classifiers LR:" + args.nb_class * "{:.1e}, ").format(*tuple(classifiers_lr)))
    else:
        classifiers_lr = [minlr]

    lr_sampler = lr_sampler_generic(minlr, maxlr)
    lr_preclassifier = generator_randomlr_neurons(net.preclassifier, lr_sampler)

    if args.optimizer == 'SGD':
        optimizer = SGDAlrao(net.parameters_preclassifier(),
                             lr_preclassifier,
                             net.classifiers_parameters_list(),
                             classifiers_lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = AdamAlrao(net.parameters_preclassifier(),
                              lr_preclassifier,
                              net.classifiers_parameters_list(),
                              classifiers_lr)

else:
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=base_lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr)


# Training
def train(epoch):
    net.train()
    if args.use_alrao:
        optimizer.update_posterior(net.posterior())
        net.switch.reset_cl_perf()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(total = data_train_size - (data_train_size % batch_size),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Epoch %d" % epoch)

    for i in range(data_train_size // batch_size):
        inputs = train_inputs[(i * batch_size):((i + 1) * batch_size)]
        targets = train_targets[(i * batch_size):((i + 1) * batch_size)]
        optimizer.zero_grad()
        # inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)
        #print(outputs)

        loss = criterion_add(outputs, targets)
        loss.backward()

        if args.use_alrao:
            optimizer.classifiers_zero_grad()
            newx = net.last_x.detach()
            for classifier in net.classifiers():
                loss_classifier = criterion(classifier(newx), targets)
                if (not remove_non_numerical) or is_numerical(loss_classifier): 
                    loss_classifier.backward()

        optimizer.step()
        train_loss += loss.item()

        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain", "{:.4f}".format(train_loss/(i + 1)))])

        # if args.use_alrao:
        #     postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)
        if args.use_alrao:
            net.update_switch(targets, catch_up=False)
            optimizer.update_posterior(net.posterior())

    pbar.close()

    if args.use_alrao:
        cl_perf = net.switch.get_cl_perf()
        for k in range(len(cl_perf)):
            print("Classifier {}\t LossTrain:{:.6f}".format(
                k, cl_perf[k]))

    return train_loss / (i + 1)


def test(epoch):
    global best_acc
    net.eval()
    if args.use_alrao:
        net.switch.reset_cl_perf()
    test_loss = 0
    for i in range(data_test_size // batch_size):
        inputs = test_inputs[(i * batch_size):((i + 1) * batch_size)]
        targets = test_targets[(i * batch_size):((i + 1) * batch_size)]
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion_add(outputs, targets)

        test_loss += loss.item()

    print('\tLossTest: %.9f' % (test_loss/(i + 1)))
    if args.use_alrao:
        print(("Posterior : "+"{:.1e}, " * args.nb_class).format(*net.posterior()))

    return test_loss / (i + 1)


t_init = time.time()
if args.early_stopping:
    earlystopping = EarlyStopping('min', patience=20)

for epoch in range(args.epochs):
    train_loss = train(epoch)
    #print('Validation')
    #valid_nll, valid_acc = test(epoch, validloader)
    print('Test')
    test_loss = test(epoch)

    """
    if args.early_stopping:
        earlystopping.step(valid_nll)
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
    """
