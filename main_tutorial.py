'''Train CIFAR with PyTorch.'''
from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from tqdm import tqdm
import numpy as np

from alrao import AlraoModel
from alrao import SGDAlrao, AdamAlrao
from alrao import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights

# CUDA
use_cuda = torch.cuda.is_available()

# Data: CIFAR-10
batch_size = 32

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Pre-classifier class
class VGG(nn.Module): # identical to models.VGG
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        # The dimension of the preclassier's output need to be specified.
        self.linearinputdim = 512

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # The model do not contain a classifier layer.
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

preclassifier = VGG([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                     512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])

# Classifier class
class Classifier(nn.Module): # identical to alrao.mymodels.LinearClassifier
    def __init__(self, in_features, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

# We define the interval in which the learning rates are sampled
minlr = 10 ** (-5)
maxlr = 10 ** 1

# nb_classifiers is the number of classifiers averaged by Alrao.
nb_classifiers = 10
nb_categories = 10
net = AlraoModel(preclassifier, nb_classifiers, Classifier, preclassifier.linearinputdim, nb_categories)
if use_cuda: net.cuda()

# We spread the classifiers learning rates log-uniformly on the interval.
classifiers_lr = [np.exp(np.log(minlr) + \
    k /(nb_classifiers-1) * (np.log(maxlr) - np.log(minlr)) \
    ) for k in range(nb_classifiers)]

# We define the sampler for the preclassifier’s features.
lr_sampler = lr_sampler_generic(minlr, maxlr)
lr_preclassifier = generator_randomlr_neurons(net.preclassifier, lr_sampler)

# We define the optimizer
optimizer = SGDAlrao(net.parameters_preclassifier(),
                     lr_preclassifier,
                     net.classifiers_parameters_list(),
                     classifiers_lr)
criterion = nn.NLLLoss()

def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(total=len(trainloader.dataset),bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Epoch %d" % epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net.train()
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # We update the model averaging weights in the optimizer
        optimizer.update_posterior(net.posterior())
        optimizer.zero_grad()

        # Forward pass of the Alrao model
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # We compute the gradient of all the model’s weights
        loss.backward()

        # We reset all the classifiers gradients, and re-compute them with
        # as if their were the only output of the network.
        optimizer.classifiers_zero_grad()
        newx = net.last_x.detach()
        for classifier in net.classifiers():
            loss_classifier = criterion(classifier(newx), targets)
            loss_classifier.backward()

        # Then, we can run an update step of the gradient descent.
        optimizer.step()

        # Finally, we update the model averaging weights
        net.update_switch(targets, catch_up=False)

        # Update loss
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progression bar
        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain","{:.4f}".format(train_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total))])
        postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)
    pbar.close()

    # Print performance of the classifiers
    cl_perf = net.switch.get_cl_perf()
    for k in range(len(cl_perf)):
        print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
            k, cl_perf[k][0], cl_perf[k][1]))

def test(epoch):
    net.eval()
    net.switch.reset_cl_perf()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        net.eval()
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        # Forward pass of the Alrao model
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # Update loss
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('\tLossTest: %.4f\tAccTest: %.3f' % (test_loss/(batch_idx+1), 100.*correct/total))
    print(("Posterior : "+"{:.1e}, " * nb_classifiers).format(*net.posterior()))

    return test_loss / (batch_idx + 1), correct / total

for epoch in range(50):
    train(epoch)
    test(epoch)
