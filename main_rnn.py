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
from alrao.alrao_model import AlraoModel
from alrao.custom_layers import LinearClassifierRNN
from alrao.optim_spec import SGDAlrao, AdamAlrao, init_alrao_optimizer, alrao_step
from alrao.learningratesgen import lr_sampler_generic, generator_randomlr_neurons, generator_randomlr_weights
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
parser.add_argument('--lr', type=float, default=1.,
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
parser.add_argument('--char_prediction', action='store_true', default=False,
                    help = 'task: character prediction')

# Alrao Parameters
parser.add_argument('--use_alrao', action='store_true', default=False,
                    help='multiple learning rates')
parser.add_argument('--minlr', type=float, default=.001,
                    help='minimum LR in alrao (eta_min)')
parser.add_argument('--maxlr', type=float, default=100.,
                    help='maximum LR in alrao (eta_max)')
parser.add_argument('--n_last_layers', type=int, default=1,
                    help='number of last layers before the switch')
parser.add_argument('--catch_up', type=int, default=-1,
                    help='catch-up period')

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
    def __init__(self, internal_nn, classifier, *args, **kwargs):
        super(StandardModel, self).__init__()
        self.internal_nn = internal_nn
        self.classifier = classifier(*args, **kwargs) #.to(device)

    def forward(self, *args, **kwargs):
        x = self.internal_nn(*args, **kwargs)

        if isinstance(x, tuple):
            x_0 = self.classifier(x[0])
            return (x_0,) + x[1:]
        else:
            return self.classifier(x)

internal_nn = RNNModel(model_name, ntokens, args.emsize, args.nhid,
                         args.nlayers, args.drop_out) #.to(device)
if args.use_alrao:
    net = AlraoModel('classification', nn.NLLLoss(), 
            internal_nn, args.n_last_layers, LinearClassifierRNN, 
            args.nhid, ntokens)
    total_param = sum(np.prod(p.size()) for p in net.parameters_internal_nn())
    total_param += sum(np.prod(p.size()) \
                       for lcparams in net.last_layers_parameters_list() \
                       for p in lcparams)
    print("Number of parameters : {:.3f}M".format(total_param / 1000000))
else:
    net = StandardModel(internal_nn, LinearClassifierRNN, args.nhid, ntokens)
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
    optimizer = init_alrao_optimizer(net, args.n_last_layers, minlr, maxlr, 
            optim_name = args.optimizer, momentum = args.momentum, weight_decay = args.weight_decay)
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
        #optimizer.update_posterior(net.posterior()) # useless beacause this action is already performed in 'alrao_step'
        net.switch.reset_ll_perf() # useless when save_ll_perf is False

    total_loss = 0.
    epoch_loss = 0.
    correct = 0
    total_pred = 0
    start_time = time.time()
    current, hidden = net.internal_nn.init_hidden(batch_size)

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

        torch.nn.utils.clip_grad_norm_(net.internal_nn.parameters(), args.clip)
        if args.use_alrao:
            catch_up = (args.catch_up != -1 and batch_idx % args.catch_up == 0)
            alrao_step(net, optimizer, criterion, targets, catch_up = catch_up, remove_non_numerical = True)
        else:
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

    pbar.close()

    if args.use_alrao:
        ll_perf = net.switch.get_ll_perf()
        for k in range(len(ll_perf)):
            print("Classifier {}\t LossTrain:{:.6f}\tAccTrain:{:.4f}".format(
                k, ll_perf[k][0], ll_perf[k][1]))

    return epoch_loss / (batch_idx + 1), correct / total_pred

def test(epoch, loader):
    net.eval()
    if args.use_alrao:
        net.switch.reset_ll_perf()
    total_loss = 0.
    correct = 0
    total_pred = 0
    ntokens = len(corpus.dictionary)
    hidden, current = net.internal_nn.init_hidden(eval_batch_size)
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
        print(("Posterior : "+"{:.1e}, " * args.n_last_layers).format(*net.posterior()))

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
