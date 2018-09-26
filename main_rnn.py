'''Train PTB with PyTorch.'''
from __future__ import print_function
from collections import namedtuple, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data

import os
import argparse
import time
from tqdm import tqdm
import numpy as np

from models import *
from mymodels import LinearClassifierRNN
from switch import Switch
from optim_spec import SGDSwitch, SGDSpec, AdamSpec, AdamSwitch
from learningratesgen import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights

from earlystopping import EarlyStopping
from alrao_model import AlraoModel
import data_text
# TO BE REMOVED
from utils import Subset
from input import parseArgs
from output import OutputManager


import pdb

from utils import *

#torch.manual_seed(123)

args = parseArgs()

outputManager = OutputManager(args)

if args.no_cuda or not torch.cuda.is_available():
    use_cuda = False
else:
    use_cuda = True

device = torch.device("cuda" if use_cuda else "cpu")

best_acc = 0  # best test accuracy

batch_size = args.rnn_batch_size
corpus = data_text.Corpus(args.rnn_data_path, raw = False, ignore_unique = True)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, batch_size)
valid_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

class StandardModel(nn.Module):
    def __init__(self, preclassifier, classifier, *args, **kwargs):
        super(StandardModel, self).__init__()
        self.preclassifier = preclassifier
        self.classifier = classifier(*args, **kwargs).to(device)

    def forward(self, x):
        x = self.preclassifier(x)
        return self.classifier(x)

"""
class StandardModel(nn.Module):
    def __init__(self, preclassifier, K=1):
        super(StandardModel, self).__init__()
        self.preclassifier = preclassifier
        self.classifier = LinearClassifier(self.preclassifier.linearinputdim, 10)

    def forward(self, x):
        x = self.preclassifier(x)
        return self.classifier(x)
"""

base_lr = args.lr
minlr = 10 ** args.minLR
maxlr = 10 ** args.maxLR

ntokens = len(corpus.dictionary)
print('Number of tokens: ' + repr(ntokens))

preclassifier = RNNModel(args.rnn_type, ntokens, args.rnn_emsize, args.rnn_nhid,
                         args.rnn_nlayers, args.dropOut).to(device)
if args.use_switch:
    net = AlraoModel(preclassifier, args.nb_class, LinearClassifierRNN, args.rnn_nhid, ntokens)
    total_param = sum(np.prod(p.size()) for p in net.parameters_preclassifier())
    total_param += sum(np.prod(p.size()) \
                       for lcparams in net.classifiers_parameters_list() \
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    net = StandardModel(preclassifier, LinearClassifierRNN)
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

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# Training
log_interval = 200
def train(epoch):
    # Turn on training mode which enables dropout.
    net.train()
    if args.use_switch:
        optimizer.update_posterior(net.posterior())
        net.switch.reset_cl_perf()
    total_loss = 0.
    epoch_loss = 0.
    nb_it = 0
    correct = 0
    total_pred = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    current, hidden = net.preclassifier.init_hidden(args.batch_size)
    for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        nb_it += 1
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden.detach_()
        current.detach_()
        net.zero_grad()
        output, hidden, current = net(data, hidden, current)
        loss = criterion(output, targets)
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

        _, predicted = torch.max(output.data, 1)
        total_pred += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(net.preclassifier.parameters(), args.clip)
        optimizer.step()

        #if args.use_switch:
        #    postfix["PostSw"] = net.repr_posterior()

        total_loss += loss.item()
        epoch_loss += loss.item()

        if args.use_switch:
            net.update_switch(targets, catch_up=batch_idx % 20 == 0)
            optimizer.update_posterior(net.posterior())

        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, len(train_data) // args.bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    if args.use_switch:
        cl_perf = net.switch.get_cl_perf()
        for k in range(len(cl_perf)):
            print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
                k, cl_perf[k][0], cl_perf[k][1]))

    return epoch_loss / nb_it, correct / total_pred

def test(epoch, loader):
    global best_acc
    # Turn on evaluation mode which disables dropout.
    net.eval()
    if args.use_switch:
        net.switch.reset_cl_perf()
    total_loss = 0.
    correct = 0
    total_pred = 0
    ntokens = len(corpus.dictionary)
    hidden, current = init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, loader.size(0) - 1, args.bptt):
            data, targets = get_batch(loader, i)
            if args.use_cuda:
                data, targets = data.cuda(), targets.cuda()
            output, hidden, current = net(data, hidden, current)
            total_loss += len(data) * criterion(output, targets).item()

            _, predicted = torch.max(output.data, 1)
            total_pred += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

    print('\tLossTest: %.4f\tAccTest: %.3f' % (total_loss / len(loader), 100. * correct / total_pred))
    if args.use_switch:
        print(("Posterior : "+"{:.1e}, " * args.nb_class).format(*net.posterior()))

    return total_loss / len(loader), correct / total_pred


t_init = time.time()
if args.early_stopping:
    earlystopping = EarlyStopping('min', patience=20)

for epoch in range(args.epochs):
    train_nll, train_acc = train(epoch)
    print('Validation')
    valid_nll, valid_acc = test(epoch, valid_data)
    print('Test')
    test_nll, test_acc = test(epoch, test_data)
    outputManager.updateData(epoch, round(time.time() - t_init), \
                             train_nll, train_acc, \
                             valid_nll, valid_acc, \
                             test_nll, test_acc)
    if args.early_stopping:
        earlystopping.step(valid_nll)
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
