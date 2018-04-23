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
from tqdm import tqdm
import numpy as np

from models import VGGNet, LinearClassifier
from switch import Switch
from optim_spec import SGDSwitch, SGDSpec, generator_lr

import pdb

torch.manual_seed(123)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

batch_size = 64
# TODO : Verifier aue le switch et ses updates se passent bien avec les mini-batchs

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class BigModel(nn.Module):
    def __init__(self, nclassifiers):
        super(BigModel, self).__init__()
        K = 6
        self.model = VGGNet(K)
        self.switch = Switch(nclassifiers)        
        self.nclassifiers = nclassifiers
        
        for i in range(nclassifiers):
            classifier = LinearClassifier(K*512, 10)
            setattr(self, "classifier"+str(i), classifier)
            
    def forward(self, x):#, y=None):
        x = self.model(x)
        lst_px = [getattr(self, "classifier"+str(i))(x) \
                  for i in range(self.nclassifiers)]
        ## WARNING : IS THIS REALLY WHAT WE WANT TO DO ?????
        #if y is not None:
        #    self.switch.Supdate(lst_px, y)
        self.last_lst_px = lst_px
        output = self.switch.forward(lst_px)
        output = output.log()
        return output

    def update_switch(self, y, x=None):
        if x is None:
            lst_px = self.last_lst_px
        else:
            lst_px = [getattr(self, "classifier"+str(i))(x) \
                  for i in range(self.nclassifiers)]
        self.switch.Supdate(lst_px, y)
            

    def parameters_model(self):
        return self.model.parameters()
        

    def classifiers_parameters_list(self):
        clparamlist = []
        for i in range(self.nclassifiers):
            classifier = getattr(self, "classifier"+str(i))
            clparamlist.append(classifier.parameters())
        return clparamlist

    def posterior(self):
        return self.switch.posterior

    def repr_posterior(self):
        post = self.switch.posterior
        bars = u' ▁▂▃▄▅▆▇█'
        res = "|"+"".join(bars[int(px)] for px in post/post.max() * 8) + "|"
        assert len(res) == self.nclassifiers + 2
        return res


class StandardModel(nn.Module):
    def __init__(self):
        super(StandardModel, self).__init__()
        self.model = VGGNet()
        self.classifier = LinearClassifier(512, 10)

    def forward(self, x):
        x = self.model(x)
        out = self.classifier(x)
        return out

base_lr = .01
minlr = 0.0001
maxlr = .1

def lr_sampler(tensor):
    """
    Takes a torch tensor as input and sample a tensor with same size for 
    learning rates
    """
    
    lr = tensor.new(tensor.size()).uniform_()
    lr = (lr * (np.log(maxlr) - np.log(minlr)) + np.log(minlr)).exp()
    #lr.fill_(base_lr)
    return lr


use_switch = True
if use_switch:
    nclassifiers = 10
    net = BigModel(nclassifiers)
else:
    net = StandardModel()
    
if use_cuda:
    net.cuda()



criterion = nn.NLLLoss()

if use_switch:
    #classifiers_lr = [base_lr for k in range(nclassifiers)]
    classifiers_lr = [np.exp(np.log(minlr) + k * (np.log(maxlr) - np.log(minlr))/nclassifiers) \
                      for k in range(nclassifiers)]
    lr_model = generator_lr(net.model, lr_sampler)
    
    optimizer = SGDSwitch(net.parameters_model(),
                          lr_model,
                          net.classifiers_parameters_list(),
                          classifiers_lr)
else:
    optimizer = optim.SGD(net.parameters(), lr=base_lr)
    
    
    
# Training
def train(epoch):
    net.train()
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
        optimizer.step()
        
        

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        pbar.update(batch_size)
        postfix = OrderedDict([("LossTrain","{:.4f}".format(train_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total)),
                               ("PostSw",net.repr_posterior())])
        pbar.set_postfix(postfix)
        if use_switch:
            net.update_switch(targets)
            optimizer.update_posterior(net.posterior())
    pbar.close()
        
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

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('\tLossTest: %.4f\tAccTest: %.3f' % (test_loss/(batch_idx+1), 100.*correct/total))
    if use_switch:
        print(("Posterior : "+"{:.3f}, " * nclassifiers).format(*net.posterior()))
    

for epoch in range(10000):
    train(epoch)
    test(epoch)



