'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from tqdm import tqdm
import numpy as np

from models import *
from mymodels import LinearClassifier
from switch import Switch
from optim_spec import SGDSwitch, SGDSpec, AdamSpec, AdamSwitch
from learningratesgen import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights

from earlystopping import EarlyStopping
from alrao_model import AlraoModel
# TO BE REMOVED
from utils import Subset

from input import parseArgs
from output import OutputManager


import pdb

from utils import *

#torch.manual_seed(123)

args = parseArgs()

outputManager = OutputManager(args)

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
trainset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=True,
                                        download=True, transform=transform_train)
trainset = Subset(trainset, list(range(0, 40000)))
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Validation set
validset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=True,
                                        download=True, transform=transform_test)
validset = Subset(validset, list(range(40000, 50000)))
validloader = data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)

# Test set
testset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=False,
                                       download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def build_preclassifier(model_name, *args, **kwargs):
    if model_name == "VGG16":
        return VGG('VGG16')
    elif model_name == "VGG19":
        return VGG('VGG19')
    elif model_name == "GoogLeNet":
        return GoogLeNet(*args, **kwargs)
    elif model_name == "MobileNetV2":
        return MobileNetV2(*args, **kwargs)
    elif model_name == "DPN92":
        return DPN92(*args, **kwargs)
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
if args.use_switch:

    net = AlraoModel(preclassifier, args.nb_class, LinearClassifier, preclassifier.linearinputdim, 10)
    total_param = sum(np.prod(p.size()) for p in net.parameters_preclassifier())
    total_param += sum(np.prod(p.size()) \
                       for lcparams in net.classifiers_parameters_list() \
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    net = StandardModel(preclassifier)
    total_param = sum(np.prod(p.size()) for p in net.parameters())
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))

if use_cuda:
    net.cuda()


criterion = nn.NLLLoss()


if args.use_switch:
    if args.nb_class > 1:
        classifiers_lr = [np.exp(\
                np.log(minlr) + k /(args.nb_class-1) * (np.log(maxlr) - np.log(minlr)) \
                ) for k in range(args.nb_class)]
        print(("Classifiers LR:" + args.nb_class * "{:.1e}, ").format(*tuple(classifiers_lr)))
    else:
        classifiers_lr = [minlr]

    lr_sampler = lr_sampler_generic(minlr, maxlr)
    lr_preclassifier = generator_randomlr_neurons(net.preclassifier, lr_sampler)

    if args.optimizer == 'SGD':
        optimizer = SGDSwitch(net.parameters_preclassifier(),
                              lr_preclassifier,
                              net.classifiers_parameters_list(),
                              classifiers_lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = AdamSwitch(net.parameters_preclassifier(),
                               lr_preclassifier,
                               net.classifiers_parameters_list(),
                               classifiers_lr)

else:
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=base_lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr)


# Training
def train(epoch, save_switch=False):
    net.train()
    if args.use_switch:
        optimizer.update_posterior(net.posterior())
        net.switch.reset_cl_perf()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(total=len(trainloader.dataset),bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Epoch %d" % epoch)

    if save_switch:
        f = open('switch_weights_e{}.txt'.format(epoch), 'w')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)


        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()


        if args.use_switch:
            optimizer.classifiers_zero_grad()
            newx = net.last_x.detach()
            for classifier in net.classifiers():
                loss_classifier = criterion(classifier(newx), targets)
                loss_classifier.backward()
                from math import isnan
                if any(np.any(np.isnan(p.grad.data)) for p in classifier.parameters()):
                    print("loss classifier:{:.5f}".format(loss_classifier.data[0]))
                    split_loss = nn.NLLLoss(reduce=False)(classifier(newx), targets)
                    maxloss = split_loss.max()
                    maxloss.backward()
                    stop


        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain","{:.4f}".format(train_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total))])
        if save_switch:
            post = net.posterior()
            f.write(';'.join('{:.4f}'.format(p) for p in post))
            f.write('\n')
        if args.use_switch:
            postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)
        if args.use_switch:
            net.update_switch(targets, catch_up=False)
            optimizer.update_posterior(net.posterior())
    if save_switch:
        f.close()
    pbar.close()

    if args.use_switch:
        cl_perf = net.switch.get_cl_perf()
        for k in range(len(cl_perf)):
            print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
                k, cl_perf[k][0], cl_perf[k][1]))

    return train_loss / (batch_idx + 1), correct / total

def test(epoch, loader):
    global best_acc
    net.eval()
    if args.use_switch:
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
    if args.use_switch:
        print(("Posterior : "+"{:.1e}, " * args.nb_class).format(*net.posterior()))

    return test_loss / (batch_idx + 1), correct / total



t_init = time.time()
if args.early_stopping:
    earlystopping = EarlyStopping('min', patience=20)

for epoch in range(args.epochs):
    train_nll, train_acc = train(epoch, save_switch=False)
    print('Validation')
    valid_nll, valid_acc = test(epoch, validloader)
    print('Test')
    test_nll, test_acc = test(epoch, testloader)
    outputManager.updateData(epoch, round(time.time() - t_init), \
                             train_nll, train_acc, \
                             valid_nll, valid_acc, \
                             test_nll, test_acc)
    if args.early_stopping:
        earlystopping.step(valid_nll)
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
