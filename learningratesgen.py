import torch
import torch.nn as nn
import numpy as np

from scipy.stats import ortho_group

import pdb


def lr_sampler_generic(minlr, maxlr):
    """
    Generates a funcion sampling random tensors according to the 'log-uniform' distribution.

    Log-uniform (LU) distribution:
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
        return lr

    return f

def generator_randomlr_neurons(module, lr_sampler, memo = None):
    """
    Generates tensors of learning rates for each submodule of the input module.
    NOTE: the learning rates are the same for each weight in a given neuron

    Arguments:
        module: the pre-classifier to be trained with Alrao
        lr_sampler: function sampling the learning rates
    """
    if memo is None:
        memo = set()

    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        memo.add(module.weight)
        w = module.weight
        lrb = lr_sampler(w, w.size()[:1])
        lrw = w.new(w.size())
        for k in range(w.size()[0]):
            lrw[k].fill_(lrb[k])
        yield lrw

        if module.bias is not None:
            memo.add(module.bias)
            yield lrb
        return

    elif isinstance(module, nn.Embedding):
        memo.add(module.weight)
        w = module.weight

        lrb = lr_sampler(w, w.size(1))
        lrw = w.new(w.size())
        lrw = lrw.t()
        for k in range(w.size(1)):
            lrw[k].fill_(lrb[k])
        lrw = lrw.t()

        yield lrw

    elif isinstance(module, nn.LSTM):
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
        return

    for _, p in module._parameters.items():
        if p is not None and p not in memo:
            print("WARNING:NOTIMPLEMENTED LAYER:{}".format(type(module)))
            memo.add(p)
            plr = lr_sampler(p, p.size())
            yield plr

    for mname, module in module.named_children():
        for lr in generator_randomlr_neurons(module, lr_sampler, memo):
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

    for mname, module in module.named_children():
        for lr in generator_randomlr_weights(module, lr_sampler, memo):
            yield lr



def generator_randomdir_neurons(module, lr_sampler, memo=None):
    if memo is None:
        memo = set()

    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) \
       or isinstance(module, nn.modules.batchnorm._BatchNorm):
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

    for mname, module in module.named_children():
        for lr in generator_randomdir_neurons(module, lr_sampler, memo):
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

    for mname, module in module.named_children():
        for lr in generator_randomdir_weights(module, lr_sampler, memo):
            yield lr
