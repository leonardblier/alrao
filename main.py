'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from tqdm import tqdm
import numpy as np

from models import VGGNet, LinearClassifier
from switch import Switch
from optim_spec import SGDSwitch, SGDSpec, generator_lr
from input import parseArgs
from output import OutputManager

import pdb

#pdb.set_trace()

torch.manual_seed(123)

args = parseArgs()
outputManager = OutputManager(args)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

batch_size = 100
# TODO : Verifier aue le switch et ses updates se passent bien avec les mini-batchs

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

trainset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class BigModel(nn.Module):
    def __init__(self, nclassifiers):
        super(BigModel, self).__init__()
        self.model = VGGNet(args.size_multiplier)
        self.switch = Switch(nclassifiers)
        self.nclassifiers = nclassifiers

        for i in range(nclassifiers):
            classifier = LinearClassifier(args.size_multiplier * 512, 10)
            setattr(self, "classifier"+str(i), classifier)

    def forward(self, x):
        x = self.model(x)
        self.last_x = x
        lst_px = [cl(x) for cl in self.classifiers()]
        self.last_lst_px = lst_px
        output = self.switch.forward(lst_px)
        output = output.log()
        return output

    def update_switch(self, y, x=None):
        if x is None:
            lst_px = self.last_lst_px
        else:
            lst_px = [cl(x) for cl in self.classifiers()]
        self.switch.Supdate(lst_px, y)


    def parameters_model(self):
        return self.model.parameters()

    def classifiers(self):
        for i in range(self.nclassifiers):
            yield getattr(self, "classifier"+str(i))



    def classifiers_parameters_list(self):
        return [cl.parameters() for cl in self.classifiers()]

    def posterior(self):
        return self.switch.posterior

    def classifiers_predictions(self, x=None):
        if x is None:
            return self.last_lst_px
        x = self.model(x)
        lst_px = [cl(x) for cl in self.classifiers()]
        self.last_lst_px = lst_px
        return lst_px

    def repr_posterior(self):
        post = self.switch.posterior
        bars = u' ▁▂▃▄▅▆▇█'
        res = "|"+"".join(bars[int(px)] for px in post/post.max() * 8) + "|"
        return res


class StandardModel(nn.Module):
    def __init__(self, K=1):
        super(StandardModel, self).__init__()
        self.model = VGGNet(K)
        self.classifier = LinearClassifier(K*512, 10)

    def forward(self, x):
        x = self.model(x)
        out = self.classifier(x)
        return F.log_softmax(out, dim = 1)

base_lr = args.lr
minlr = args.minLR
maxlr = args.maxLR

def lr_sampler(tensor):
    """
    Takes a torch tensor as input and sample a tensor with same size for
    learning rates
    """

    lr = tensor.new(tensor.size()).uniform_()
    lr = (lr * (np.log(maxlr) - np.log(minlr)) + np.log(minlr)).exp()
    #lr.fill_(base_lr)
    return lr


if args.use_switch:
    net = BigModel(args.nb_class)
else:
    net = StandardModel()

if use_cuda:
    net.cuda()


criterion = nn.NLLLoss()

if args.use_switch:
    classifiers_lr = [base_lr for k in range(args.nb_class)]
    #classifiers_lr = [np.exp(np.log(minlr) + k * (np.log(maxlr) - np.log(minlr))/args.nb_class) \
    #                  for k in range(args.nb_class)]
    lr_model = generator_lr(net.model, lr_sampler)

    optimizer = SGDSwitch(net.parameters_model(),
                          lr_model,
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
    if args.use_switch:
        optimizer.update_posterior(net.posterior())
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(total=len(trainloader.dataset),bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Epoch %d" % epoch)
    #with tqdm(total=100) as pbar:

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)


        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        if args.use_switch:
            newx = Variable(net.last_x.data.clone())
            for classifier in net.classifiers():
                loss_classifier = criterion(classifier(newx).log(), targets)
                loss_classifier.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain","{:.4f}".format(train_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total))])
        if args.use_switch:
            postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)
        if args.use_switch:
            net.update_switch(targets)
            optimizer.update_posterior(net.posterior())
    pbar.close()

    return train_loss / (batch_idx + 1), correct / total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    print('\tLossTest: %.4f\tAccTest: %.3f' % (test_loss/(batch_idx+1), 100.*correct/total))
    if args.use_switch:
        print(("Posterior : "+"{:.3f}, " * args.nb_class).format(*net.posterior()))

    return test_loss / (batch_idx + 1), correct / total

t_init = time.time()
for epoch in range(args.epochs):
    train_nll, train_acc = train(epoch)
    test_nll, test_acc = test(epoch)
    outputManager.updateData(epoch, round(time.time() - t_init), \
                             train_nll, train_acc, \
                             0, 0, \
                             test_nll, test_acc)
