import sys
import argparse
from collections import defaultdict


def parseArgs():
    parser = argparse.ArgumentParser(description='weight-decay')

    # CUDA
    parser.add_argument('--no-cuda', action = 'store_true', default = False, \
                        help = 'disable cuda')

    # epochs
    parser.add_argument('--epochs', type = int, default = 1, \
                        help = 'number of epochs for phase 1 (default: 1)')
    parser.add_argument('--epochs2', type = int, default = 0, \
                        help = 'number of epochs for phase 2 (default: 0)')

    # Drop out
    parser.add_argument('--dropOut', type = float, default = 0, \
                        help = 'drop-out rate in the first phase (default: 0)')
    parser.add_argument('--dropOut2', type = float, default = 0, \
                        help = 'drop-out rate in the second phase (default: 0)')

    # prior
    parser.add_argument('--penalty', action = 'store_true', default = False, \
                        help = 'add penalty to loss')

    # options
    parser.add_argument('--optimizer', default = 'Adam', \
                        help = 'optimizer (default: Adam) {Adam, SGD, RMSprop}')
    parser.add_argument('--lr', type = float, default = .01, \
                        help = 'learning rate (default: .01)')
    parser.add_argument('--suffix', default = '', \
                        help = 'suffix for the file name')
    parser.add_argument('--exp_number', type = int, default = -1, \
                        help = 'for seq jobs: number of the run')
    parser.add_argument('--data_augm', action = 'store_true', default = False, \
                        help = 'add data augmentation')
    parser.add_argument('--use_switch', action = 'store_true', default = False, \
                        help = 'multiple learning rates')
    parser.add_argument('--minLR', type = int, default = 0, \
                        help = 'log of the minimum LR')
    parser.add_argument('--maxLR', type = int, default = 0, \
                        help = 'log of the maximum LR')
    parser.add_argument('--nb_class', type = int, default = 0, \
                        help = 'number of layers before the switch')
    parser.add_argument('--stats', action = 'store_true', default = False, \
                        help = 'outputs PDF with stats over weights')
    parser.add_argument('--size_multiplier', type = float, default = 1, \
                        help = 'multiplier of the number of neurons per layer (default: 1)')

    # print options
    parser.add_argument('--print_period', type = int, default = 10, \
                        help = 'sort period (default: 10)')
    parser.add_argument('--print_TWeights', action = 'store_true', default = False, \
                        help = 'periodic output of TWeights files')

    # batchnorm
    parser.add_argument('--batchnorm', default = 'None', \
                        help = 'batchnorm (default: None) {None, BB, BN, BNP}')
    parser.add_argument('--BN_lr', type = float, default = .1, \
                        help = 'batchnorm learning rate (default: .1)')
    parser.add_argument('--BN_affine', action = 'store_true', default = False, \
                        help = 'activate linear layer after batchnorm')

    args = parser.parse_args()

    return args
