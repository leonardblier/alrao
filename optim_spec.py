import os
from collections import defaultdict, OrderedDict
import math
import torch
import torch.nn as nn
import torch.optim as optim

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

class AdamSpec:
    def __init__(self, named_parameters, named_lr, lr, betas = (.9, .999), eps = 1e-8):
        self.learning_rate = lr
        self.named_parameters = OrderedDict(named_parameters)
        self.named_lr = named_lr
        self.beta1, self.beta2 = betas
        self.state = defaultdict(dict)
        self.epsilon = eps

    def step(self):
        for p1, p2 in self.named_parameters.items():
            if p2.grad is None:
                continue
            grad = p2.grad.data

            #state = self.state[p2]
            state = self.state[p1]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(self.beta1).add_(1 - self.beta1, grad)
            exp_avg_sq.mul_(self.beta2).addcmul_(1 - self.beta2, grad, grad)

            denom = exp_avg_sq.sqrt().add_(self.epsilon)

            bias_correction1 = 1 - self.beta1 ** state['step']
            bias_correction2 = 1 - self.beta2 ** state['step']

            if p1 in self.named_lr:
                step_size = math.sqrt(bias_correction2) / bias_correction1
                wd = self.named_lr[p1]
                p2.data.addcdiv_(-step_size, exp_avg.mul(wd), denom)
            else:
                step_size = self.learning_rate * math.sqrt(bias_correction2) / bias_correction1
                p2.data.addcdiv_(-step_size, exp_avg, denom)

    def zero_grad(self):
        for p1, p2 in self.named_parameters.items():
            if p2.grad is not None:
                if p2.grad.volatile:
                    p2.grad.data.zero_()
                else:
                    data = p2.grad.data
                    p2.grad = Variable(data.new().resize_as_(data).zero_())

class SGDSpec(optim.Optimizer):
    def __init__(self, named_parameters, gen_lr):
        # TODO : Remove all the names
        self.named_parameters = OrderedDict(named_parameters)
        
        self.named_lr = OrderedDict()
        for name, p in self.named_parameters.items():
            p_lr = gen_lr(p)
            self.named_lr[name] = p_lr   

    def step(self):
        # TODO : Remove all the names
        for (namep,p),(namelr, wlr) in zip(self.named_parameters.items(), self.named_lr.items()):
            assert(namep == namelr)
            if p.grad is None:
                continue
            grad = p.grad.data
            p.data.addcmul_(-1, grad, wlr)

    def zero_grad(self):
        for name, p in self.named_parameters.items():
            if p.grad is not None:
                if p.grad.volatile:
                    p.grad.data.zero_()
                else:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())


class SGDSwitch(optim.Optimizer):
    def __init__(self, named_parameters_model, classifiers_parameters_list,
                 classifiers_lr, gen_lr):

        self.sgdmodel = SGDSpec(named_parameters_model, gen_lr)
        self.classifiers_lr = classifiers_lr
        self.sgdclassifiers = [optim.SGD(parameters, lr) for parameters, lr in \
                               zip(classifiers_parameters_list, classifiers_lr)]

    def update_posterior(self, posterior):
        self.posterior = posterior


    def step(self):
        self.sgdmodel.step()
        for sgdclassifier, posterior, lr in zip(self.sgdclassifiers,
                                                self.posterior, self.classifiers_lr):
            for param_group in sgdclassifier.param_groups:
                param_group['lr'] = lr / posterior
            sgdclassifier.step()
            
        
    def zero_grad(self):
        self.sgdmodel.zero_grad()
        for opt in self.sgdclassifiers:
            opt.zero_grad()

    
