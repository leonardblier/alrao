'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import argparse
import time
from tqdm import tqdm
import numpy as np

from models import GoogLeNet, MobileNetV2, VGG, SENet18
from mymodels import LinearClassifier
from optim_spec import SGDAlrao, AdamAlrao
from learningratesgen import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights

from earlystopping import EarlyStopping
from alrao_model import AlraoModel
# TO BE REMOVED
from utils import Subset


parser = argparse.ArgumentParser(description='alrao')

# CUDA
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable cuda')

# epochs
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs for phase 1 (default: 50)')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='use early stopping')

# options
parser.add_argument('--model_name', default='GoogLeNet',
                    help='Model {VGG19, GoogLeNet, MobileNetV2, SENet18}')
parser.add_argument('--optimizer', default='SGD',
                    help='optimizer (default: SGD) {Adam, SGD}')
parser.add_argument('--lr', type=float, default=.01,
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
parser.add_argument('--minLR', type=int, default=-5,
                    help='log10 of the minimum LR in alrao (log_10 eta_min)')
parser.add_argument('--maxLR', type=int, default=0,
                    help='log10 of the maximum LR in alrao (log_10 eta_max)')
parser.add_argument('--nb_class', type=int, default=10,
                    help='number of classifiers before the switch')

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

batch_size = 32

# Data
print('==> Preparing data..')
if args.data_augm:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Train set
DATA_DIR = './datasets'
trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=True, transform=transform_train)
trainset = Subset(trainset, list(range(0, 40000)))
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Validation set
validset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                        download=True, transform=transform_test)
validset = Subset(validset, list(range(40000, 50000)))
validloader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)

# Test set
testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                       download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def build_preclassifier(model_name, *args, **kwargs):
    if model_name == "VGG19":
        return VGG('VGG19')
    elif model_name == "GoogLeNet":
        return GoogLeNet(*args, **kwargs)
    elif model_name == "MobileNetV2":
        return MobileNetV2(*args, **kwargs)
    elif model_name == "SENet18":
        return SENet18(*args, **kwargs)
    else:
        raise ValueError("Unknown model name : {}".format(model_name))


class StandardModel(nn.Module):
    def __init__(self, preclassifier, K=1):
        super(StandardModel, self).__init__()
        self.preclassifier = preclassifier
        self.classifier = LinearClassifier(self.preclassifier.linearinputdim, 10)

    def forward(self, x):
        x = self.preclassifier(x)
        return self.classifier(x)


base_lr = args.lr
minlr = 10 ** args.minLR
maxlr = 10 ** args.maxLR

preclassifier = build_preclassifier(args.model_name, gamma=args.size_multiplier)
if args.use_alrao:

    net = AlraoModel(preclassifier, args.nb_class, LinearClassifier,
                     preclassifier.linearinputdim, 10)
    total_param = sum(np.prod(p.size()) for p in net.parameters_preclassifier())
    total_param += sum(np.prod(p.size())
                       for lcparams in net.classifiers_parameters_list()
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    net = StandardModel(preclassifier)
    total_param = sum(np.prod(p.size()) for p in net.parameters())
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))

if use_cuda:
    net.cuda()


criterion = nn.NLLLoss()


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

    pbar = tqdm(total=len(trainloader.dataset),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Epoch %d" % epoch)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        # inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        if args.use_alrao:
            optimizer.classifiers_zero_grad()
            newx = net.last_x.detach()
            for classifier in net.classifiers():
                loss_classifier = criterion(classifier(newx), targets)
                loss_classifier.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain", "{:.4f}".format(train_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total))])

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
            print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
                k, cl_perf[k][0], cl_perf[k][1]))

    return train_loss / (batch_idx + 1), correct / total


def test(epoch, loader):
    global best_acc
    net.eval()
    if args.use_alrao:
        net.switch.reset_cl_perf()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('\tLossTest: %.4f\tAccTest: %.3f' % (test_loss/(batch_idx+1), 100.*correct/total))
    if args.use_alrao:
        print(("Posterior : "+"{:.1e}, " * args.nb_class).format(*net.posterior()))

    return test_loss / (batch_idx + 1), correct / total


t_init = time.time()
if args.early_stopping:
    earlystopping = EarlyStopping('min', patience=20)

for epoch in range(args.epochs):
    train_nll, train_acc = train(epoch)
    print('Validation')
    valid_nll, valid_acc = test(epoch, validloader)
    print('Test')
    test_nll, test_acc = test(epoch, testloader)

    if args.early_stopping:
        earlystopping.step(valid_nll)
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
