import os
from collections import defaultdict, OrderedDict
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

                 
class AdamSpec(optim.Optimizer):
    def __init__(self, params, lr_params, betas = (.9, .999), eps = 1e-8,
                 weight_decay=0, amsgrad=False):
        
        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(AdamSpec, self).__init__(params, defaults)

        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]
            
        # self.learning_rate = lr
        # self.named_parameters = OrderedDict(named_parameters)
        # self.named_lr = named_lr
        # self.beta1, self.beta2 = betas
        # self.state = defaultdict(dict)
        # self.epsilon = eps

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p, lr_p in zip(group["params"], group["lr_list"]):
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
            
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                    
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg * lr_p, denom)



def generator_lr(module, lr_sampler, memo=None):
    if memo is None:
        memo = set()

    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        memo.add(module.weight)
        memo.add(module.bias)
        
        lrb = lr_sampler(module.bias.data)
        w = module.weight.data
        lrw = w.new(w.size())
        for k in range(w.size()[0]):
            lrw[k].fill_(lrb[k])
        yield lrw
        yield lrb
        return
        
    for _, p in module._parameters.items():
        if p is not None and p not in memo:
            print("WARNING:NOTIMPLEMENTED LAYER:{}".format(type(module)))
            memo.add(p)
            plr = lr_sampler(p.data)
            yield plr
            
    for mname, module in module.named_children():
        for lr in generator_lr(module, lr_sampler, memo):
            yield lr


class SGDSpec(optim.Optimizer):
    def __init__(self, params, lr_params, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        # TODO : Remove all the names
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

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                d_p = p.grad.data
                p.data.addcmul_(-1., d_p, lr_p)
                

    
class SGDSwitch:
    def __init__(self, parameters_model, lr_model, classifiers_parameters_list,
                 classifiers_lr):
        #super(SGDSwitch, self).__init__()
        self.sgdmodel = SGDSpec(parameters_model, lr_model)
        self.classifiers_lr = classifiers_lr
        self.sgdclassifiers = [optim.SGD(parameters, lr) for parameters, lr in \
                               zip(classifiers_parameters_list, classifiers_lr)]
        # self.sgdclassifiers = [optim.Adam(parameters, lr) for parameters, lr in \
        #                        zip(classifiers_parameters_list, classifiers_lr)]

    def update_posterior(self, posterior):
        self.posterior = posterior


    def step(self):
        self.sgdmodel.step()    
        for sgdclassifier, posterior, lr in zip(self.sgdclassifiers,
                                                self.posterior,
                                                self.classifiers_lr):
            for param_group in sgdclassifier.param_groups:
                #param_group['lr'] = lr / posterior
                sgdclassifier.step()

    def zero_grad(self):
        self.sgdmodel.zero_grad()
        for opt in self.sgdclassifiers:
            opt.zero_grad()

    
