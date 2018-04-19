import os
import argparse
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.nn.modules.loss import _WeightedLoss
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from util import *
from switch import *

# Parser
args = parseArgs()
use_cuda = torch.cuda.is_available() and not args.no_cuda

# Size of the neural network (VGG-13)
#net_sizes = [3, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 10] # principal
net_sizes = [3, 64, 64, 128, 128, 256, 256, 512, 512, 512, 512, 10] # temoin

# Pre-processing of the arguments (args)
if args.nb_neur_mult != 1: # Modify the number of neurons by layer
    for i in range(1, len(net_sizes) - 1):
        net_sizes[i] = int(net_sizes[i] * args.nb_neur_mult)

if not args.mLR:
    args.minLR = math.log10(args.lr)
    args.maxLR = args.minLR

# Hyperparameters
input_size = 32 * 32 # size of an image
dimChannel = 1 # size of the channels before the fully connected layer
train_set_size = 50000 # size of the loaded train set
valid_size = 8000 # number of images taken from the train set to build the validation set
train_size = train_set_size - valid_size # size of the real train set
test_size = 10000 # size of the test set
batch_size = 100

mMom = True

# Stats over the weights
nbStat = 4000 # size of the sample (per layer)
statEpsilon = 1e-8 # below this value, a weight is considered as equal to zero

KL_factor = 1 / train_size # factor of the KL-penalty

# dict that contains all the hyperparameters about the batchnorm layers
bnArgs = {'type': args.batchnorm, \
          'affine': args.BN_affine, \
          'momentum': args.BN_lr, \
          'step_mult': batch_size / train_size}
epsilon = args.negl_neurons

# Init output
nameBase = genNameBase(args)
nameBase += '_h' + repr(net_sizes[1])
initIncludes(nameBase)
InitJournal(nameBase)
InitList(nameBase)

# CIFAR Dataset
if args.data_augm:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding = 4), \
                                          transforms.RandomHorizontalFlip(), \
                                          transforms.ToTensor()])
else:
    transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])

train_dataset = dsets.CIFAR10(root = '../data_CIFAR10', train = True,
                              download = True, transform = transform_train)

test_dataset = dsets.CIFAR10(root = '../data_CIFAR10', train = False,
                             download = True, transform = transform_test)

train_loader, train_loader_seq, valid_loader, test_loader = \
    buildLoaders(train_dataset, test_dataset, batch_size, valid_size)

classes = ('plane', 'car', 'bird', 'cat', \
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Define the way the hyperparameters are generated
f_lr = fdet_lr_log(args.minLR, args.maxLR) if args.mLR else fdet_lr_unif # learning rate
f_mom = fdet_mom_id # momentum (unused)

# Neural Network Model

   
            
    
        
class Net(nn.Module):
    def __init__(self, net_sizes, nb_switch, args = None):
        super(Net, self).__init__()

        self.sizes = net_sizes
        self.nb_switch = nb_switch
        self.activeDO = False

        # Layers
        self.createLayer('conv00', nn.Conv2d(self.sizes[0], self.sizes[1], 3, padding = 1), f_lr, f_mom)
        self.createLayer('conv01', nn.Conv2d(self.sizes[1], self.sizes[2], 3, padding = 1), f_lr, f_mom)

        self.createLayer('conv10', nn.Conv2d(self.sizes[2], self.sizes[3], 3, padding = 1), f_lr, f_mom)
        self.createLayer('conv11', nn.Conv2d(self.sizes[3], self.sizes[4], 3, padding = 1), f_lr, f_mom)

        self.createLayer('conv20', nn.Conv2d(self.sizes[4], self.sizes[5], 3, padding = 1), f_lr, f_mom)
        self.createLayer('conv21', nn.Conv2d(self.sizes[5], self.sizes[6], 3, padding = 1), f_lr, f_mom)

        self.createLayer('conv30', nn.Conv2d(self.sizes[6], self.sizes[7], 3, padding = 1), f_lr, f_mom)
        self.createLayer('conv31', nn.Conv2d(self.sizes[7], self.sizes[8], 3, padding = 1), f_lr, f_mom)

        self.createLayer('conv40', nn.Conv2d(self.sizes[8], self.sizes[9], 3, padding = 1), f_lr, f_mom)
        self.createLayer('conv41', nn.Conv2d(self.sizes[9], self.sizes[10], 3, padding = 1), f_lr, f_mom)

        #self.createLayer('fc', nn.Linear(self.sizes[10] * dimChannel, self.sizes[11]), f_lr_unif)

        self.createSwitch()

        # Drop-out
        self.setDropOut(args.dropOut)

        # Batchnorm
        self.typeBN = 'None' if args == None else args.batchnorm

        if bnArgs['type'] == 'BN':
            self.bnCV00 = nn.BatchNorm2d(self.sizes[0], affine = bnArgs['affine'], momentum = bnArgs['momentum'])
            self.bnCV01 = nn.BatchNorm2d(self.sizes[1], affine = bnArgs['affine'], momentum = bnArgs['momentum'])

            self.bnCV10 = nn.BatchNorm2d(self.sizes[2], affine = bnArgs['affine'], momentum = bnArgs['momentum'])
            self.bnCV11 = nn.BatchNorm2d(self.sizes[3], affine = bnArgs['affine'], momentum = bnArgs['momentum'])

            self.bnCV20 = nn.BatchNorm2d(self.sizes[4], affine = bnArgs['affine'], momentum = bnArgs['momentum'])
            self.bnCV21 = nn.BatchNorm2d(self.sizes[5], affine = bnArgs['affine'], momentum = bnArgs['momentum'])

            self.bnCV30 = nn.BatchNorm2d(self.sizes[6], affine = bnArgs['affine'], momentum = bnArgs['momentum'])
            self.bnCV31 = nn.BatchNorm2d(self.sizes[7], affine = bnArgs['affine'], momentum = bnArgs['momentum'])

            self.bnCV40 = nn.BatchNorm2d(self.sizes[8], affine = bnArgs['affine'], momentum = bnArgs['momentum'])
            self.bnCV41 = nn.BatchNorm2d(self.sizes[9], affine = bnArgs['affine'], momentum = bnArgs['momentum'])

            self.bnFC = nn.BatchNorm1d(self.sizes[10] * dimChannel, affine = bnArgs['affine'], momentum = bnArgs['momentum'])

    def testModel(self, test_loader, use_cuda, test_size = 0, setName = '', criterion = None):
            self.train(False)
            loss_nll, eff = testModel(self, test_loader, use_cuda, test_size, setName, criterion)
            self.train(True)
            return loss_nll, eff

    def setDropOut(self, p):
        if p != 0:
            self.activeDO = True
            self.doCV01 = nn.Dropout2d(p = p / 1)

            self.doCV10 = nn.Dropout2d(p = p)
            self.doCV11 = nn.Dropout2d(p = p)

            self.doCV20 = nn.Dropout2d(p = p)
            self.doCV21 = nn.Dropout2d(p = p)

            self.doCV30 = nn.Dropout2d(p = p)
            self.doCV31 = nn.Dropout2d(p = p)

            self.doCV40 = nn.Dropout2d(p = p)
            self.doCV41 = nn.Dropout2d(p = p)

            self.doFC = nn.Dropout2d(p = p)
        else:
            self.activeDO = False

    def createLayer(self, layerName, layer, f_lr, f_mom = 0):
        if not hasattr(self, 'named_lr'):
            self.named_lr = {}
            #self.named_mom = {}

        setattr(self, layerName, layer)

        if isinstance(f_lr, float) or isinstance(f_lr, int):
            self.named_lr[layerName + '.weight'] = fdet_lr_unif(layer.weight.data, f_lr).cuda()
        else:
            self.named_lr[layerName + '.weight'] = f_lr(layer.weight.data).cuda()

        """
        if isinstance(f_mom, float) or isinstance(f_mom, int):
            self.named_mom[layerName + '.weight'] = f_mom_unif(layer.weight.data).cuda()
        else:
            self.named_mom[layerName + '.weight'] = f_mom_id(layer.weight.data).cuda()
        """

    def createSwitch(self):
        r"""
        Builds self.nb_switch fully connected output layers and an instance of
            Switch, that keeps a posterior distribution over these
        """
        self.switch = Switch(self.nb_switch)

        if self.nb_switch == 1:
            self.createLayer('fc0', nn.Linear(self.sizes[10] * dimChannel, self.sizes[11]), fdet_lr_unif)
        else:
            log_lr_dst = args.maxLR - args.minLR
            for i in range(self.nb_switch):
                log_lr = args.minLR + log_lr_dst * (i / (self.nb_switch - 1))
                lr = pow(10, log_lr)
                includes.outputText('Switch FC #' + repr(i) + ': lr = ' + repr(lr))
                self.createLayer('fc' + repr(i), nn.Linear(self.sizes[10] * dimChannel, self.sizes[11]), lr)

    def loss(self):
        ret = 0

        ret += self.conv00.weight.pow(2).sum()
        ret += self.conv01.weight.pow(2).sum()
        ret += self.conv10.weight.pow(2).sum()
        ret += self.conv11.weight.pow(2).sum()
        ret += self.conv20.weight.pow(2).sum()
        ret += self.conv21.weight.pow(2).sum()
        ret += self.conv30.weight.pow(2).sum()
        ret += self.conv31.weight.pow(2).sum()
        ret += self.conv40.weight.pow(2).sum()
        ret += self.conv41.weight.pow(2).sum()
        for i in range(self.nb_switch):
            ret += getattr(self, 'fc' + repr(i)).weight.pow(2).sum()

        return ret

    def update_switch(self, y):
        self.switch.Supdate(self.lst_x, y)

    def fwd_switch(self, x):
        self.lst_x = [None] * self.nb_switch
        for i in range(self.nb_switch):
            fc = getattr(self, 'fc' + repr(i))
            self.lst_x[i] = F.log_softmax(fc(x), dim = 1)

        return self.switch(self.lst_x)

    
    def forward(self, x):
        if bnArgs['type'] != 'None': x = self.bnCV00(x)
        x = self.conv00(x)
        x = F.relu(x)
        if bnArgs['type'] != 'None': x = self.bnCV01(x)
        if self.activeDO: x = self.doCV01(x)
        x = self.conv01(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride = 2)

        if bnArgs['type'] != 'None': x = self.bnCV10(x)
        if self.activeDO: x = self.doCV10(x)
        x = self.conv10(x)
        x = F.relu(x)
        if bnArgs['type'] != 'None': x = self.bnCV11(x)
        if self.activeDO: x = self.doCV11(x)
        x = self.conv11(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride = 2)

        if bnArgs['type'] != 'None': x = self.bnCV20(x)
        if self.activeDO: x = self.doCV20(x)
        x = self.conv20(x)
        x = F.relu(x)
        if bnArgs['type'] != 'None': x = self.bnCV21(x)
        if self.activeDO: x = self.doCV21(x)
        x = self.conv21(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride = 2)

        if bnArgs['type'] != 'None': x = self.bnCV30(x)
        if self.activeDO: x = self.doCV30(x)
        x = self.conv30(x)
        x = F.relu(x)
        if bnArgs['type'] != 'None': x = self.bnCV31(x)
        if self.activeDO: x = self.doCV31(x)
        x = self.conv31(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride = 2)

        if bnArgs['type'] != 'None': x = self.bnCV40(x)
        if self.activeDO: x = self.doCV40(x)
        x = self.conv40(x)
        x = F.relu(x)
        if bnArgs['type'] != 'None': x = self.bnCV41(x)
        if self.activeDO: x = self.doCV41(x)
        x = self.conv41(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride = 2)

        x = x.view(x.size(0), -1)
        if bnArgs['type'] != 'None': x = self.bnFC(x)
        if self.activeDO: x = self.doFC(x)

        return self.fwd_switch(x)

def getStatLayer(l):
    tab = l.weight.data.cpu()
    tab.resize_(min(tab.nelement(), nbStat))
    tab = tab.numpy()
    tab_keep = np.logical_or(tab > statEpsilon, -statEpsilon > tab)
    n = tab.size
    nb_nonzero = np.sum(tab_keep)
    nb_zero = n - nb_nonzero

    stat_nonzero = np.zeros(nb_nonzero)
    j = 0
    for i in range(n):
        if tab_keep[i]:
            stat_nonzero[j] = tab[i]
            j += 1

    return nb_zero, stat_nonzero

def printStatLayer(l, lName, StatFile, epoch):
    nb_zero, stat_nonzero = getStatLayer(l)
    plt.hist(stat_nonzero, bins = 'auto')
    plt.title('Stats ' + lName + ', epoch ' + repr(epoch) + ', zeros: ' + repr(nb_zero) + '/' + repr(stat_nonzero.size + nb_zero))
    StatFile.savefig()
    plt.clf()

def printStatAll(net, StatFile, epoch):
    printStatLayer(net.conv00, 'conv00', StatFile, epoch)
    printStatLayer(net.conv01, 'conv01', StatFile, epoch)
    printStatLayer(net.conv10, 'conv10', StatFile, epoch)
    printStatLayer(net.conv11, 'conv11', StatFile, epoch)
    printStatLayer(net.conv20, 'conv20', StatFile, epoch)
    printStatLayer(net.conv21, 'conv21', StatFile, epoch)
    printStatLayer(net.conv30, 'conv30', StatFile, epoch)
    printStatLayer(net.conv31, 'conv31', StatFile, epoch)
    printStatLayer(net.conv40, 'conv40', StatFile, epoch)
    printStatLayer(net.conv41, 'conv41', StatFile, epoch)
    printStatLayer(net.fc, 'fc', StatFile, epoch)

net = Net(net_sizes, args.switch, args)

if use_cuda:
    net = net.cuda()

# Loss and Optimizer
criterion = nn.NLLLoss()
if args.optimizer == 'Adam':
    optimizer = AdamSpec(net.named_parameters(), net.named_lr, lr = args.lr)
elif args.optimizer == 'SGD':
    optimizer = SGDSpec(net.named_parameters(), net.named_lr, lr = args.lr)
else:
    print('Error: optimizer unrecognized', file = sys.stderr)
    sys.exit()

def InitSwitch(fileSwitch, net):
    U_str = 'epoch '
    for i in range(net.nb_switch):
        U_str += '\t'
        U_str += 'sw_' + repr(i)
    fileSwitch(U_str)

def WriteSwitch(fileSwitch, net, epoch):
    U_str = format(epoch, '.4f')
    for i in range(net.nb_switch):
        U_str += '\t'
        U_str += format(net.switch.posterior[i], '.4f')
    fileSwitch(U_str)

# Monitor
monitor = Monitor(net, criterion, \
                  train_loader_seq, valid_loader, test_loader, \
                  train_size, valid_size, test_size, use_cuda)

# Train the Model
InitLossFile()
#InitNeurFile(net)
InitPata(net)
fileSwitch = outputTextConstructor('Switch_' + nameBase + '.txt')
InitSwitch(fileSwitch, net)
KL_factor_spec = .5 * KL_factor
bn_start = True
t_init = time.time()
for epoch in range(args.epochs + args.epochs2):
    if epoch == args.epochs:
        if args.stats:
            StatFile = PdfPages(os.getcwd() + '/Stat_' + nameBase + '_e' + repr(epoch) + '.pdf')
            printStatAll(net, StatFile, epoch)
            StatFile.close()

        net.setDropOut(args.dropOut2)

        args.penalty = False

        """
        if optimName == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr = learning_rate)
        elif optimName == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr = lr_adam)
        elif optimName == 'RMSprop':
            optimizer = optim.RMSprop(net.parameters(), lr = lr_rmsprop)
        """

    cum_train_pen = 0
    cum_loss_nll = 0
    cum_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images)
        labels = Variable(labels)
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_nll = loss.clone()

        if args.penalty:
            current_pen = KL_factor_spec * net.loss()
            loss += current_pen
            cum_train_pen += current_pen.data[0]

        cum_loss_nll += loss_nll.data[0]
        cum_loss += loss.data[0]

        loss.backward()
        optimizer.step()
        net.update_switch(labels)

        if (i + 1) % 100 == 0:
            includes.outputText('Epoch [%d], Step [%d/%d], NLL_Loss: %.4f, Loss: %.4f'
                                %(epoch + 1, i + 1, train_size // batch_size,
                                loss_nll.data[0], loss.data[0]))
            WriteSwitch(fileSwitch, net, epoch + (batch_size * i / train_size))

    # Ending operations
    if valid_loader != None:
        monitor.updateValid(nameBase, epsilon, epoch + 1)
        WriteLoss(cum_train_pen / i, cum_loss_nll / i, cum_loss / i, \
                  KL_factor_spec * monitor.valid_pen, monitor.valid_nll, monitor.valid_eff, \
                  KL_factor_spec * monitor.best_test_pen, monitor.best_test_nll, monitor.best_test_eff)
    """
    if periodPrint != 0:
        if (epoch + 1) % periodPrint == 0:
            net.writeNorms(nameBase, epsilon, epoch + 1, file = args.print_TWeights)
    WriteNbNeurons(net)
    """

    # Gros patapouf
    WritePata(epoch + 1, int(round((time.time() - t_init) * 1000)), net, \
              cum_train_pen / i, cum_loss_nll / i, cum_loss / i, \
              KL_factor_spec * monitor.valid_pen, monitor.valid_nll, monitor.valid_eff, \
              KL_factor_spec * monitor.best_test_pen, monitor.best_test_nll, monitor.best_test_eff)


    # Preparation for the next epoch
    """
    if epoch < args.epochs + args.epochs2 - 1:
        if args.prune_perm != -1: net.pruneRaw(args.prune_perm, optimizer)
    """

# Test the model
includes.outputText('Best accuracy of the network on the %d test images: %.2f %%' \
                    %(test_size, 100 * monitor.best_test_eff))
monitor.finalTest()
#net.writeNorms(nameBase, epsilon)
if args.stats:
    StatFile = PdfPages(os.getcwd() + '/Stat_' + nameBase + '_e' + repr(epoch + 1) + '.pdf')
    printStatAll(net, StatFile, epoch + 1)
    StatFile.close()

# Save the Model
torch.save(net.state_dict(), 'model_' + nameBase + '.pkl')
