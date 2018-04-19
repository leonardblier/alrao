'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

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



from models import VGGNet, LinearClassifier
from switch import Switch
from optim_spec import SGDSwitch

import pdb

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/titanic_1/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class BigModel(nn.Module):
    def __init__(self, nclassifiers):
        super(BigModel, self).__init__()
        self.model = VGGNet()
        self.switch = Switch(nclassifiers)
        self.nclassifiers = nclassifiers
        
        for i in range(nclassifiers):
            classifier = LinearClassifier(512, 10)
            setattr(self, "classifier"+str(i), classifier)
            
    def forward(self, x, y=None):
        x = self.model(x)
        lst_x = []
        for i in range(self.nclassifiers):
            classifier = getattr(self, "classifier"+str(i))
            lst_x.append(classifier(x))

        ## WARNING : IS THIS REALLY WHAT WE WANT TO DO ?????
        if y is not None:
            self.switch.Supdate(lst_x, y)
            
        output = self.switch.forward(lst_x)

        return output

    def named_parameters_model(self):
        return self.model.named_parameters()

    def classifiers_parameters_list(self):
        clparamlist = []
        for i in range(self.nclassifiers):
            classifier = getattr(self, "classifier"+str(i))
            clparamlist.append(classifier.parameters())
        return clparamlist

    def posterior(self):
        return self.switch.posterior


class StandardModel(nn.Module):
    def __init__(self):
        super(StandardModel, self).__init__()
        self.model = VGGNet()
        self.classifier = LinearClassifier(512, 10)

    def forward(self, x):
        x = self.model(x)
        out = self.classifier(x)
        return out

    
def gen_lr(p):
    if use_cuda:
        lr = torch.cuda.FloatTensor(p.data)
    else:
        lr = torch.Tensor(p.data)
    #lr.uniform_()
    #lr = lr * 20. - 15.
    #lr = torch.pow(2., lr)
    lr.fill_(.1)
    return lr


use_switch = False
if use_switch:
    nclassifiers = 1
    net = BigModel(nclassifiers)
else:
    net = StandardModel()
    
if use_cuda:
    net.cuda()



criterion = nn.CrossEntropyLoss()

if use_switch:
    #classifiers_lr = [2. ** (5 - k) for k in range(nclassifiers)]
    classifiers_lr = [0.1 for k in range(nclassifiers)]
    optimizer = SGDSwitch(net.named_parameters_model(),
                          net.classifiers_parameters_list(),
                          classifiers_lr,
                          gen_lr)
else:
    optimizer = SGD(net.parameters(), lr=.1)
    
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        
        if use_switch:
            outputs = net(inputs, y=targets)
            optimizer.update_posterior(net.posterior())
        else:
            outputs = net(inputs)
            
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

        print("Test")
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    

for epoch in range(10000):
    train(epoch)
    test(epoch)



