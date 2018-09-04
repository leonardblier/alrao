import os
from collections import defaultdict, OrderedDict
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy.stats import ortho_group

import pdb
r"""
These optimizers are copies of the original pytorch optimizers with one change:
    They take 2 specific arguments: named_parameters (instead of params) and named_lr.
    While looping over the dictionary named_parameters, these optimizers check whether
    the name of the current layer is in named_lr. If not, the optimization is
    exaclty the same as in the original optimizer. If it is the case, it looks
    into named_lr and take the tensor of specific learning rates associated to
    this layer, then uses it to update the layer.
"""


def generator_randomlr_neurons(module, lr_sampler, memo=None):
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

                 
class AdamSpec(optim.Optimizer):
    def __init__(self, params, lr_params, betas = (.9, .999), eps = 1e-8,
                 weight_decay=0, amsgrad=False):

        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(AdamSpec, self).__init__(params, defaults)

        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, lr_p in zip(group['params'], group['lr_list']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, torch.mul(exp_avg, lr_p), denom)
        return loss




class SGDSpec(optim.Optimizer):
    def __init__(self, params, lr_params, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGDSpec, self).__init__(params, defaults)

        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]


    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, lr_p in zip(group['params'], group['lr_list']):
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # Warning : Are we sure we want to clamp ?
                        buf.addcmul_(-momentum, lr_p.sqrt().clamp(0., 1/momentum), buf)
                        buf.add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.addcmul_(-1., d_p, lr_p)
                


class SGDRandDirNeuronsSpec(optim.Optimizer):
    def __init__(self, params, lr_params):
        defaults = dict()
        super(SGDRandDirNeuronsSpec, self).__init__(params, defaults)
        
        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]
            

    def __setstate__(self, state):
        super(SGDRandDirNeuronsSpec, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, lr_p in zip(group['params'], group['lr_list']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                dim = d_p.dim()
                if dim == 1:
                    p.data.addmv_(lr_p, d_p, alpha=-1)
                elif dim == 2:
                    p.data.addmm_(d_p, lr_p, alpha=-1)
                elif dim >= 3:
                    update = torch.matmul(lr_p, d_p.transpose(0, -2)).transpose(0,-2)
                    p.data.add_(-1, update)
                else:
                    raise ValueError


class SGDRandDirWeightsSpec(optim.Optimizer):
    def __init__(self, params, lr_params):
        defaults = dict()
        super(SGDRandDirWeightsSpec, self).__init__(params, defaults)
        
        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]
            

    def __setstate__(self, state):
        super(SGDRandDirWeightsSpec, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, lr_p in zip(group['params'], group['lr_list']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                size = d_p.size()

                update = torch.mv(lr_p, d_p.view(-1))
                p.data.add(-1, update.view(size))


                
                
class OptSwitch:
    def __init__(self):
        pass

    def update_posterior(self, posterior):
        self.posterior = posterior

    def step(self):
        if self.optmodel is not None:
            self.optmodel.step()
        for optclassifier in self.optclassifiers:
            optclassifier.step()
        # for sgdclassifier, posterior, lr in zip(self.sgdclassifiers,
        #                                         self.posterior,
        #                                         self.classifiers_lr):
        #     for param_group in sgdclassifier.param_groups:
        #         #param_group['lr'] = lr / posterior
        #         sgdclassifier.step()

    def classifiers_zero_grad(self):
        for opt in self.optclassifiers:
            opt.zero_grad()

    def zero_grad(self):
        if self.optmodel is not None:
            self.optmodel.zero_grad()
        for opt in self.optclassifiers:
            opt.zero_grad()


class SGDSwitch(OptSwitch):
    def __init__(self, parameters_model, lr_model, classifiers_parameters_list,
                 classifiers_lr, momentum=0., weight_decay=0.):

        super(SGDSwitch, self).__init__()
        self.optmodel = SGDSpec(parameters_model, lr_model,
                                momentum=momentum, weight_decay=weight_decay)
        self.classifiers_lr = classifiers_lr
        self.optclassifiers = \
            [optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay) \
             for parameters, lr in zip(classifiers_parameters_list, classifiers_lr)]


class AdamSwitch(OptSwitch):
    def __init__(self, parameters_model, lr_model, classifiers_parameters_list,
                 classifiers_lr, **kwargs):

        super(AdamSwitch, self).__init__()
        #self.classifiers_lr = classifiers_lr

        self.optmodel = AdamSpec(parameters_model, lr_model, **kwargs)
        self.optclassifiers = [optim.Adam(parameters, lr, **kwargs) \
             for parameters, lr in zip(classifiers_parameters_list, classifiers_lr)]
