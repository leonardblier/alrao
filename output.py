import argparse
import math
import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import Optimizer


# Format and output data
def genNameBase(args):
    nameBase = ''

    addUnderscore = False
    if args.penalty:
        nameBase += 'p'
        addUnderscore = True
    if args.batchnorm != 'None':
        nameBase += args.batchnorm
        addUnderscore = True
    if args.data_augm:
        nameBase += 'da'
        addUnderscore = True

    if addUnderscore:
        nameBase += '_'

    addUnderscore = False
    if args.dropOut != 0:
        nameBase += 'do-'
        nameBase += format(args.dropOut, '.3f')
        addUnderscore = True

    if addUnderscore:
        nameBase += '_'

    nameBase += 'lr-'
    nameBase += format(args.lr, '.4f')

    if args.minLR != args.maxLR and args.use_switch:
        nameBase += '_mLR-'
        nameBase += 'p' if args.minLR >= 0 else 'm'
        nameBase += repr(abs(args.minLR))
        nameBase += 'p' if args.maxLR >= 0 else 'm'
        nameBase += repr(abs(args.maxLR))

    if args.nb_class != 0:
        nameBase += '_sw-'
        nameBase += repr(args.nb_class)

    nameBase += '_' + args.optimizer
    nameBase += '_e' + repr(args.epochs)

    if args.exp_number != -1:
        nameBase += '_exp-'
        nameBase += repr(args.exp_number)

    if args.suffix != '':
        nameBase += '_' + args.suffix

    return nameBase

# Journal of experiments
def InitJournal(nameBase):
    includes.outputGlobal('------------' + str(datetime.datetime.now()) + '------------')
    includes.outputGlobal(nameBase)
    includes.outputGlobal('Loss_' + nameBase + '.txt')

    includes.outputGlobal('============ run.sh ============')
    with open("run.py") as f:
        lines = f.readlines()
        for l in lines:
            includes.outputGlobal(l.rstrip())
    includes.outputGlobal('================================')

    includes.outputGlobal('\n')

# Journal of experiments
def InitList(nameBase):
    includes.outputList(nameBase)

# Output loss
def InitLossFile():
    includes.outputLoss('train_pen  ' + \
                        '\t' + 'train_nll' + \
                        '\t' + 'train_loss' + \
                        '\t' + 'valid_pen' + \
                        '\t' + 'valid_nll' + \
                        '\t' + 'valid_eff' + \
                        '\t' + 'test_pen' + \
                        '\t' + 'test_nll' + \
                        '\t' + 'test_eff')

def WriteLoss(train_pen, train_nll, train_loss, \
              valid_pen, valid_nll, valid_eff, \
              test_pen, test_nll, test_eff):
    includes.outputLoss('%.4f   \t%.4f   \t%.4f   \t%.4f   \t%.4f   \t%.4f   \t%.4f   \t%.4f   \t%.4f' \
                        %(train_pen, train_nll, train_loss, \
                          valid_pen, valid_nll, valid_eff, \
                          test_pen, test_nll, test_eff))

# Output grosPatapouf
def InitPata(net):
    U_str = 'epoch'
    U_str += '\t' + 'time'

    U_str += '\t' + 'tot_neur'
    if hasattr(net, 'listLayers'):
        for layer in net.listLayers:
            U_str += '\t' + layer.name2

    U_str += '\t' + 'train_pen'
    U_str += '\t' + 'train_nll'
    U_str += '\t' + 'train_loss'
    U_str += '\t' + 'valid_pen'
    U_str += '\t' + 'valid_nll'
    U_str += '\t' + 'valid_eff'
    U_str += '\t' + 'test_pen'
    U_str += '\t' + 'test_nll'
    U_str += '\t' + 'test_eff'

    includes.grosPatapouf(U_str)

def WritePata(epoch, t, net, \
              train_pen, train_nll, train_loss, \
              valid_pen, valid_nll, valid_eff, \
              test_pen, test_nll, test_eff):
    U_str = ''
    total = 0
    if hasattr(net, 'listLayers'):
        for layer in net.listLayers:
            total += layer.weight.data.size()[0]
            U_str += '\t' + repr(layer.weight.data.size()[0])
    U_str = repr(total) + U_str

    U_str = repr(epoch) + '\t' + repr(t) + '\t' + U_str

    U_str += '\t' + repr(train_pen)
    U_str += '\t' + repr(train_nll)
    U_str += '\t' + repr(train_loss)
    U_str += '\t' + repr(valid_pen)
    U_str += '\t' + repr(valid_nll)
    U_str += '\t' + repr(valid_eff)
    U_str += '\t' + repr(test_pen)
    U_str += '\t' + repr(test_nll)
    U_str += '\t' + repr(test_eff)

    includes.grosPatapouf(U_str)
