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

    if args.use_switch and args.nb_class != 0:
        nameBase += '_sw-'
        nameBase += repr(args.nb_class)

    nameBase += '_' + args.optimizer
    nameBase += '_K' + args.size_multiplier
    nameBase += '_e' + repr(args.epochs)

    if args.exp_number != -1:
        nameBase += '_exp-'
        nameBase += repr(args.exp_number)

    if args.suffix != '':
        nameBase += '_' + args.suffix

    return nameBase

class OutputManager:
    def __init__(self, args):
        self.nameBase = genNameBase(args)

        fileLstNames = open('FileNamesList.txt', 'a')
        fileLstNames.write(self.nameBase + '\n')
        fileLstNames.close()

        self.nameFileData = 'Data_' + self.nameBase + '.txt'
        self.initData()

    def initData(self):
        fileData = open(self.nameFileData, 'w')
        U_str = 'epoch'
        U_str += '\t' + 'time'
        U_str += '\t' + 'train_nll'
        U_str += '\t' + 'train_acc'
        U_str += '\t' + 'valid_nll'
        U_str += '\t' + 'valid_acc'
        U_str += '\t' + 'test_nll'
        U_str += '\t' + 'test_acc'
        fileData.write(U_str + '\n')
        fileData.close()

    def updateData(self, epoch, t, \
                   train_nll, train_acc, \
                   valid_nll, valid_acc, \
                   test_nll, test_acc):
        fileData = open(self.nameFileData, 'a')
        U_str = repr(epoch)
        U_str += '\t' + repr(t)
        U_str += '\t' + "{:.4f}".format(train_nll)
        U_str += '\t' + "{:.5f}".format(train_acc)
        U_str += '\t' + "{:.4f}".format(valid_nll)
        U_str += '\t' + "{:.5f}".format(valid_acc)
        U_str += '\t' + "{:.4f}".format(test_nll)
        U_str += '\t' + "{:.5f}".format(test_acc)
        fileData.write(U_str + '\n')
        fileData.close()
