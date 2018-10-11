'''Train PTB with PyTorch.'''
from __future__ import print_function
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import time
from tqdm import tqdm
import numpy as np

from models import RNNModel
from alrao import AlraoModel, LinearClassifierRNN
from alrao import SGDAlrao, AdamAlrao
from alrao import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights
from alrao.earlystopping import EarlyStopping

import data.data_text as data_text


parser = argparse.ArgumentParser(description='alrao')

# CUDA
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable cuda')

# epochs
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs for phase 1 (default: 50)')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='use early stopping')

# options
parser.add_argument('--drop_out', type=float, default=.2,
                    help='drop-out rate')
parser.add_argument('--optimizer', default='SGD',
                    help='optimizer (default: SGD) {Adam, SGD}')
parser.add_argument('--lr', type=float, default=.01,
                    help='learning rate, when used without alrao')
parser.add_argument('--momentum', type=float, default=0.,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='l2 penalty')

# RNN
parser.add_argument('--data_path', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--clip', type=float, default=.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--char_prediction', action = 'store_true', default = False,
                    help = 'task: character prediction')

# Alrao Parameters
parser.add_argument('--use_alrao', action='store_true', default=True,
                    help='multiple learning rates')
parser.add_argument('--minlr', type=float, default=.001,
                    help='minimum LR in alrao (eta_min)')
parser.add_argument('--maxlr', type=float, default=100.,
                    help='maximum LR in alrao (eta_max)')
parser.add_argument('--nb_class', type=int, default=6,
                    help='number of classifiers before the switch')

args = parser.parse_args()

if args.no_cuda or not torch.cuda.is_available():
    use_cuda = False
else:
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model_name = 'LSTM'

# Build dataset
batch_size = args.batch_size
eval_batch_size = 10

corpus = data_text.Corpus(args.data_path, char_prediction = args.char_prediction)
ntokens = len(corpus.dictionary)
print('Number of tokens: ' + repr(ntokens))

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, batch_size)
valid_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build model
class StandardModel(nn.Module):
    def __init__(self, preclassifier, classifier, *args, **kwargs):
        super(StandardModel, self).__init__()
        self.preclassifier = preclassifier
        self.classifier = classifier(*args, **kwargs) #.to(device)

    def forward(self, *args, **kwargs):
        x = self.preclassifier(*args, **kwargs)

        z = x
        if isinstance(z, tuple):
            z = x[0]

        out = self.classifier(z)

        if isinstance(x, tuple):
            out = (out,) + x[1:]
        return out

preclassifier = RNNModel(model_name, ntokens, args.emsize, args.nhid,
                         args.nlayers, args.drop_out) #.to(device)
if args.use_alrao:
    net = AlraoModel(preclassifier, args.nb_class, LinearClassifierRNN, args.nhid, ntokens)
    total_param = sum(np.prod(p.size()) for p in net.parameters_preclassifier())
    total_param += sum(np.prod(p.size()) \
                       for lcparams in net.classifiers_parameters_list() \
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    net = StandardModel(preclassifier, LinearClassifierRNN, args.nhid, ntokens)
    total_param = sum(np.prod(p.size()) for p in net.parameters())
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))

if use_cuda:
    net.cuda()

criterion = nn.NLLLoss()

# Build optimizer
base_lr = args.lr
minlr = args.minlr
maxlr = args.maxlr

if args.use_alrao:
    # Build the learning rates for each classifier
    if args.nb_class > 1:
        classifiers_lr = [np.exp(\
                np.log(minlr) + k /(args.nb_class-1) * (np.log(maxlr) - np.log(minlr)) \
                ) for k in range(args.nb_class)]
        print(("Classifiers LR:" + args.nb_class * "{:.1e}, ").format(*tuple(classifiers_lr)))
    else:
        classifiers_lr = [minlr]

    # Build the learning rates for the pre-classifier
    lr_sampler = lr_sampler_generic(minlr, maxlr)
    lr_preclassifier = generator_randomlr_neurons(net.preclassifier, lr_sampler)

    if args.optimizer == 'SGD':
        optimizer = SGDAlrao(net.parameters_preclassifier(),
                              lr_preclassifier,
                              net.classifiers_parameters_list(),
                              classifiers_lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = AdamAlrao(net.parameters_preclassifier(),
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
    net.train()
    if args.use_alrao:
        optimizer.update_posterior(net.posterior())
        net.switch.reset_cl_perf()
    total_loss = 0.
    epoch_loss = 0.
    correct = 0
    total_pred = 0
    start_time = time.time()
    current, hidden = net.preclassifier.init_hidden(batch_size)

    pbar = tqdm(total = train_data.size(0),
                bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
    pbar.set_description("Epoch %d" % epoch)

    for batch_idx, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        data, targets = data.cuda(), targets.cuda()

        hidden.detach_()
        current.detach_()
        net.zero_grad()
        output, hidden, current = net(data, hidden, current)
        loss = criterion(output, targets)
        loss.backward()

        if args.use_alrao:
            optimizer.classifiers_zero_grad()
            newx = net.last_x.detach()
            for classifier in net.classifiers():
                loss_classifier = criterion(classifier(newx), targets)
                loss_classifier.backward()

        torch.nn.utils.clip_grad_norm_(net.preclassifier.parameters(), args.clip)
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total_pred += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        total_loss += loss.item()
        epoch_loss += loss.item()

        pbar.update(args.bptt)
        postfix = OrderedDict([("LossTrain", "{:.4f}".format(total_loss/(batch_idx+1))),
                               ("AccTrain", "{:.3f}".format(100.*correct/total_pred))])
        if args.use_alrao:
            postfix["PostSw"] = net.repr_posterior()
        pbar.set_postfix(postfix)

        if args.use_alrao:
            net.update_switch(targets, catch_up=batch_idx % 20 == 0)
            optimizer.update_posterior(net.posterior())

    pbar.close()

    if args.use_alrao:
        cl_perf = net.switch.get_cl_perf()
        for k in range(len(cl_perf)):
            print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
                k, cl_perf[k][0], cl_perf[k][1]))

    return epoch_loss / (batch_idx + 1), correct / total_pred

def test(epoch, loader):
    net.eval()
    if args.use_alrao:
        net.switch.reset_cl_perf()
    total_loss = 0.
    correct = 0
    total_pred = 0
    ntokens = len(corpus.dictionary)
    hidden, current = net.preclassifier.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, loader.size(0) - 1, args.bptt):
            data, targets = get_batch(loader, i)
            if use_cuda:
                data, targets = data.cuda(), targets.cuda()
            output, hidden, current = net(data, hidden, current)
            total_loss += len(data) * criterion(output, targets).item()

            _, predicted = torch.max(output.data, 1)
            total_pred += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

    print('\tLossTest: %.4f\tAccTest: %.3f' % (total_loss / len(loader), 100. * correct / total_pred))
    if args.use_alrao:
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
    if args.early_stopping:
        earlystopping.step(valid_nll)
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
