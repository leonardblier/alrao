"""
Tools for generating learning rates
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ortho_group



def lr_sampler_generic(minlr, maxlr):
    """
    Generates a funcion sampling random tensors according to the 'log-uniform' distribution.

    Log-uniform (LU) distribution
        Z ~ U(log(a), log(b))
        X = exp(Z)
        Y = LU(a, b)
        # X and Y have the same distribution

    Arguments:
        minlr, maxlr: bounds of the log-uniform distribution
    """
    def f(tensor, size):
        """
        Takes a torch tensor as input and sample a tensor with same size for
        learning rates.
        """
        lr = tensor.new(size).uniform_()
        lr = (lr * (np.log(maxlr) - np.log(minlr)) + np.log(minlr)).exp()
        #lr.fill_(base_lr)
        return lr.cuda()

    return f


def generator_randomlr_linconv(module, lr_sampler, memo):
    """
    Generates tensors of learning rates the parameters of Linear or convolutional layers.
    NOTE: the learning rates are the same for each weight in a given neuron

    Arguments:
        module: the linear or conv layer
        lr_sampler: function sampling the learning rates
    """
    memo.add(module.weight)
    w = module.weight
    lrb = lr_sampler(w, w.size()[:1])
    lrw = lrb.new(w.size())
    for k in range(w.size()[0]):
        lrw[k].fill_(lrb[k])
    yield lrw

    if module.bias is not None:
        memo.add(module.bias)
        yield lrb

def generator_randomlr_embedding(module, lr_sampler, memo):
    """
    Generates tensors of learning rates the parameters of Embedding layers.
    NOTE: the learning rates are the same for each weight in a given neuron

    Arguments:
        module: the embedding layer
        lr_sampler: function sampling the learning rates
    """
    memo.add(module.weight)
    w = module.weight

    lrb = lr_sampler(w, w.size(1))
    lrw = w.new(w.size())
    lrw = lrw.t()
    for k in range(w.size(1)):
        lrw[k].fill_(lrb[k])
    lrw = lrw.t()

    yield lrw

def generator_randomlr_lstm(module, lr_sampler, memo):
    """
    Generates tensors of learning rates the parameters of LSTM layers.
    NOTE: the learning rates are the same for each weight in a given neuron

    Arguments:
        module: the embedding layer
        lr_sampler: function sampling the learning rates
    """
    dct_lr = {}
    for name, p in module._parameters.items():
        if name.find('weight_ih') == 0:
            memo.add(p)
            lrb_ih = lr_sampler(p, p.size()[:1])

            lrw_ih = p.new(p.size())
            for k in range(p.size(0)):
                lrw_ih[k].fill_(lrb_ih[k])
            yield lrw_ih

        elif name.find('weight_hh') == 0:
            memo.add(p)
            lrb_hh = lr_sampler(p, p.size()[:1])
            lrw_hh = p.new(p.size())
            for k in range(p.size()[0]):
                lrw_hh[k].fill_(lrb_hh[k])
            yield lrw_hh
        elif name.find('bias_ih') == 0:
            memo.add(p)
            yield lrb_ih
        elif name.find('bias_hh') == 0:
            memo.add(p)
            yield lrb_hh
        else:
            print("switch module: optim_spec.py: WARNING: UNKNOWN PARAMETER IN LSTM MODULE: {}".format(name))

def generator_randomlr_default(module, lr_sampler, memo, warning=False):
    """
    Generates tensors of learning rates, default method

    Arguments:
        module: the embedding layer
        lr_sampler: function sampling the learning rates
    """
    if warning:
        print(("Warning: The module {} does not have a specific learning rate sampler. "
               "Each weight will be sampled independantly with lr_sampler, "
               "not with a learning rate by feature.").format(type(module)))
    for _, p in module._parameters.items():
        if p is not None and p not in memo:
            memo.add(p)
            plr = lr_sampler(p, p.size())
            yield plr


def generator_randomlr_neurons(module, lr_sampler, memo=None):
    """
    Generates tensors of learning rates for each submodule of the input module.
    NOTE: the learning rates are the same for each weight in a given neuron

    Arguments:
        module: the pre-classifier to be trained with Alrao
        lr_sampler: function sampling the learning rates
    """
    if memo is None:
        memo = set()

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        gen = generator_randomlr_linconv
    elif isinstance(module, nn.Embedding):
        gen = generator_randomlr_embedding
    elif isinstance(module, nn.LSTM):
        gen = generator_randomlr_lstm
    else:
        gen = generator_randomlr_default


    for lr in gen(module, lr_sampler, memo):
        yield lr

    for child_module in module.children():
        for lr in generator_randomlr_neurons(child_module, lr_sampler, memo):
            yield lr


def generator_randomlr_weights(module, lr_sampler, memo=None):
    """
    Generates tensors of learning rates for each submodule of the input module.
    NOTE: every weight is associated to its own learning rate

    Arguments:
        module: the pre-classifier to be trained with Alrao
        lr_sampler: function sampling the learning rates
    """
    if memo is None:
        memo = set()

    for _, p in module._parameters.items():
        if p is not None and p not in memo:
            memo.add(p)
            plr = lr_sampler(p, p.size())
            yield plr

    for child_module in module.children():
        for lr in generator_randomlr_weights(child_module, lr_sampler, memo):
            yield lr



def generator_randomdir_neurons(module, lr_sampler, memo=None):
    """
    TODO
    """
    if memo is None:
        memo = set()

    if isinstance(module, (nn.Linear, nn.Conv2d, nn.modules.batchnorm._BatchNorm)):
        memo.add(module.weight)
        w = module.weight

        # THIS IS UGLY
        lrb = np.diag(lr_sampler(w.cpu(), w.size()[:1]).numpy())

        basis = ortho_group.rvs(w.size()[0])
        Hb = np.dot(np.dot(basis.T, lrb), basis)
        Hb = w.new_tensor(Hb)
        yield Hb

        if module.bias is not None:
            memo.add(module.bias)
            yield Hb
        return

    for _, p in module._parameters.items():
        if p is not None and p not in memo:
            print("WARNING:NOTIMPLEMENTED LAYER:{}".format(type(module)))
            memo.add(p)
            plr = lr_sampler(p, p.size())
            yield plr

    for child_module in module.named_children():
        for lr in generator_randomdir_neurons(child_module, lr_sampler, memo):
            yield lr


def generator_randomdir_weights(module, lr_sampler, memo=None):
    if memo is None:
        memo = set()

    for _, p in module._parameters.items():
        if p is not None and p not in memo:
            memo.add(p)

            size = int(np.prod(p.size()))
            print("Generating random directions of size {}".format(size))

            basis, _ = np.linalg.qr(np.random.randn(size, size))
            plr = np.diag(lr_sampler(p.cpu(), torch.Size([size])).numpy())
            Hb = np.dot(np.dot(basis.T, plr), basis)

            Hb = p.new_tensor(Hb)
            yield Hb

    for child_module in module.children():
        for lr in generator_randomdir_weights(child_module, lr_sampler, memo):
            yield lr
