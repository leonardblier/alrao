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
from alrao.custom_layers import LinearClassifier, LinearRegressor, L2LossLog, L2LossAdditional
from alrao.optim_spec import SGDAlrao, AdamAlrao, init_alrao_optimizer, alrao_step
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
parser.add_argument('--use_alrao', action='store_true', default=False,
                    help='multiple learning rates')
parser.add_argument('--minLR', type=int, default=-3,  # base = -5
                    help='log10 of the minimum LR in alrao (log_10 eta_min)')
parser.add_argument('--maxLR', type=int, default=-3,  # base = 0
                    help='log10 of the maximum LR in alrao (log_10 eta_max)')
parser.add_argument('--n_last_layers', type=int, default=1,
                    help='number of last layers before the switch')
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

# Model (internal NN)
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

def build_internal_nn(model_name, *args, **kwargs):
    return RegModel(*args)

class StandardModel(nn.Module):
    def __init__(self, internal_nn, K=1):
        super(StandardModel, self).__init__()
        self.internal_nn = internal_nn
        self.last_layer = LinearClassifier(self.internal_nn.linearinputdim, 10)

    def forward(self, x):
        x = self.internal_nn(x)
        return self.last_layer(x)

class StandardModelReg(nn.Module):
    def __init__(self, internal_nn):
        super(StandardModelReg, self).__init__()
        self.internal_nn = internal_nn
        self.regressor = LinearRegressor(self.internal_nn.linearinputdim, 1)

    def forward(self, x):
        x = self.internal_nn(x)
        x = self.regressor(x)
        return x.unsqueeze(2), x.new().resize_(1).fill_(1.)

base_lr = args.lr
minlr = 10 ** args.minLR
maxlr = 10 ** args.maxLR

internal_nn = build_internal_nn(args.model_name, input_dim, pre_output_dim)
criterion = L2LossLog(sigma2 = sigma2)
criterion_add = L2LossAdditional(sigma2 = sigma2)
if args.use_alrao:
    net = AlraoModel(internal_nn, args.n_last_layers, LinearRegressor,
                     'regression', criterion, internal_nn.linearinputdim, 1)
    total_param = sum(np.prod(p.size()) for p in net.parameters_internal_nn())
    total_param += sum(np.prod(p.size())
                       for lcparams in net.last_layers_parameters_list()
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    if args.task == 'classification':
        net = StandardModel(internal_nn)
    elif args.task == 'regression':
        net = StandardModelReg(internal_nn)
    total_param = sum(np.prod(p.size()) for p in net.parameters())
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))

if use_cuda:
    net.cuda()

if args.use_alrao:
    optimizer = init_alrao_optimizer(net, args.n_last_layers, minlr, maxlr, 
            optim_name = args.optimizer, momentum = args.momentum, weight_decay = args.weight_decay)
else:
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr = base_lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = base_lr)


# Training
def train(epoch):
    net.train()
    """
    if args.use_alrao:
        optimizer.update_posterior(net.posterior()) # useless beacause this action is already performed in 'alrao_step'
        net.switch.reset_ll_perf() # useless when save_ll_perf is False
    """
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

        outputs = net(inputs)

        loss = criterion_add(outputs, targets)
        loss.backward()

        if args.use_alrao:
            alrao_step(net, optimizer, criterion, targets, catch_up = False, remove_non_numerical = True) 
        else:
            optimizer.step()

        train_loss += loss.item()

        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain", "{:.4f}".format(train_loss/(i + 1)))])

        # if args.use_alrao:
        #     postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)

    pbar.close()

    if args.use_alrao:
        ll_perf = net.switch.get_ll_perf()
        for k in range(len(ll_perf)):
            print("Last layer {}\t LossTrain:{:.6f}".format(
                k, ll_perf[k]))

    return train_loss / (i + 1)


def test(epoch):
    global best_acc
    net.eval()
    if args.use_alrao:
        net.switch.reset_ll_perf()
    test_loss = 0
    for i in range(data_test_size // batch_size):
        inputs = test_inputs[(i * batch_size):((i + 1) * batch_size)]
        targets = test_targets[(i * batch_size):((i + 1) * batch_size)]

        outputs = net(inputs)
        loss = criterion_add(outputs, targets)

        test_loss += loss.item()

    print('\tLossTest: %.9f' % (test_loss/(i + 1)))
    if args.use_alrao:
        print(("Posterior : "+"{:.1e}, " * args.n_last_layers).format(*net.posterior()))

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
