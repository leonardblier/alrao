'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
from collections import namedtuple, OrderedDict

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

from models import *
from mymodels import LinearClassifier
from switch import Switch
from optim_spec import SGDSwitch, SGDSpec, generator_lr


from input import parseArgs
from output import OutputManager


import pdb

from utils import *

#torch.manual_seed(123)

args = parseArgs()

torch.manual_seed(123)

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

trainset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def build_model(model_name, *args, **kwargs):
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

class BigModel(nn.Module):
    def __init__(self, nclassifiers):
        super(BigModel, self).__init__()
        self.switch = Switch(nclassifiers, save_cl_perf=True)        
        #self.model = VGGNet(args.size_multiplier)
        self.model = build_model(args.model_name, gamma=args.size_multiplier)
        self.nclassifiers = nclassifiers

        for i in range(nclassifiers):
            classifier = LinearClassifier(self.model.linearinputdim, 10)
            setattr(self, "classifier"+str(i), classifier)

    def forward(self, x):
        x = self.model(x)
        lst_logpx = [cl(x) for cl in self.classifiers()]
        self.last_x, self.last_lst_logpx = x, lst_logpx
        return self.switch.forward(lst_logpx)

    def update_switch(self, y, x=None, catch_up=True):
        if x is None:
            lst_px = self.last_lst_logpx
        else:
            lst_px = [cl(x) for cl in self.classifiers()]
        self.switch.Supdate(lst_px, y)

        if catch_up:
            self.hard_catch_up()

    def hard_catch_up(self, threshold=-20):
        logpost = self.switch.logposterior
        weak_cl = [cl for cl, lp in zip(self.classifiers(), logpost) if lp < threshold]
        if len(weak_cl) == 0:
            return None

        mean_weight = torch.stack(
            [cl.fc.weight * p for (cl, p) in zip(self.classifiers(), logpost.exp())],
            dim=-1).sum(dim=-1).detach()
        mean_bias = torch.stack(
            [cl.fc.bias * p for (cl, p) in zip(self.classifiers(), logpost.exp())],
            dim=-1).sum(dim=-1).detach()
        for cl in weak_cl:
            cl.fc.weight.data = mean_weight.clone()
            cl.fc.bias.data = mean_bias.clone()

    def parameters_model(self):
        return self.model.parameters()

    def classifiers(self):
        for i in range(self.nclassifiers):
            yield getattr(self, "classifier"+str(i))

    def classifiers_parameters_list(self):
        return [cl.parameters() for cl in self.classifiers()]

    def posterior(self):
        return self.switch.logposterior.exp()

    def classifiers_predictions(self, x=None):
        if x is None:
            return self.last_lst_logpx
        x = self.model(x)
        lst_px = [cl(x) for cl in self.classifiers()]
        self.last_lst_logpx = lst_px
        return lst_px

    def repr_posterior(self):
        post = self.posterior()
        bars = u' ▁▂▃▄▅▆▇█'
        res = "|"+"".join(bars[int(px)] for px in post/post.max() * 8) + "|"
        return res





class StandardModel(nn.Module):
    def __init__(self, K=1):
        super(StandardModel, self).__init__()
        self.model = build_model(args.model_name, gamma=K)
        self.classifier = LinearClassifier(self.model.linearinputdim, 10)

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x)

base_lr = args.lr
minlr = 10 ** args.minLR
maxlr = 10 ** args.maxLR


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
    total_param = sum(np.prod(p.size()) for p in net.parameters_model())
    total_param += sum(np.prod(p.size()) \
                       for lcparams in net.classifiers_parameters_list() \
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    net = StandardModel(args.size_multiplier)
    total_param = sum(np.prod(p.size()) for p in net.parameters())
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))

if use_cuda:
    net.cuda()


criterion = nn.NLLLoss()


if args.use_switch:
    #classifiers_lr = [base_lr for k in range(args.nb_class)]
    classifier_minlr = 10 ** (-5)
    classifier_maxlr = 1.
    classifiers_lr = [np.exp(np.log(classifier_minlr) + k * (np.log(classifier_maxlr) - np.log(classifier_minlr))/args.nb_class) \
                      for k in range(args.nb_class)]

    lr_model = generator_lr(net.model, lr_sampler)

    optimizer = SGDSwitch(net.parameters_model(),
                          lr_model,
                          net.classifiers_parameters_list(),
                          classifiers_lr)
else:
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr)


# Training
def train(epoch):
    net.train()
    if args.use_switch:
        optimizer.update_posterior(net.posterior())
        net.switch.reset_cl_perf()
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

        # print("Before Aux Gradient:")
        # l2params(net)
        #print(net.posterior())
        if args.use_switch:
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
        # print("After Aux Gradient:")
        # l2params(net)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain","{:.4f}".format(train_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total))])
        if args.use_switch:
            postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)
        if args.use_switch:
            net.update_switch(targets, catch_up=batch_idx % 20 == 0)
            optimizer.update_posterior(net.posterior())
            

        # if args.use_switch:
        #     cl_perf = net.switch.get_cl_perf()
        #     posterior = net.posterior()
        #     for k in range(len(cl_perf)):
        #         print("Classifier {}\t Posterior {:.8f}\tLossTrain:{:.8f}\tAccTrain:{:.4f}".format(
        #             k, posterior[k], cl_perf[k][0], cl_perf[k][1]))
    pbar.close()

    if args.use_switch:
        cl_perf = net.switch.get_cl_perf()
        for k in range(len(cl_perf)):
            print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
                k, cl_perf[k][0], cl_perf[k][1]))

    return train_loss / (batch_idx + 1), correct / total

def test(epoch):
    global best_acc
    net.eval()
    if args.use_switch:
        net.switch.reset_cl_perf()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
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
        print(("Posterior : "+"{:.3f}, " * args.nb_class).format(*net.posterior()))
        # cl_perf = net.switch.get_cl_perf()
        # for k in range(len(cl_perf)):
        #     print("Classifier {}\t LossTrain:{:.4f}\tAccTrain:{:.2f}".format(
        #         k, cl_perf[k][0], cl_perf[k][1]))
    return test_loss / (batch_idx + 1), correct / total



t_init = time.time()
for epoch in range(args.epochs):
    train_nll, train_acc = train(epoch)
    test_nll, test_acc = test(epoch)
    outputManager.updateData(epoch, round(time.time() - t_init), \
                             train_nll, train_acc, \
                             0, 0, \
                             test_nll, test_acc)
