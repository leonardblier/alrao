import os
from collections import defaultdict
import math
import torch
import torch.nn as nn

r"""
These optimizers are copies of the original pytorch optimizers with one change:
    They take 2 specific arguments: named_params (instead of params) and named_lr.
    While looping over the dictionary named_params, these optimizers check whether
    the name of the current layer is in named_lr. If not, the optimization is
    exaclty the same as in the original optimizer. If it is the case, it looks
    into named_lr and take the tensor of specific learning rates associated to
    this layer, then uses it to update the layer.
"""

class AdamSpec:
    def __init__(self, named_params, named_lr, lr, betas = (.9, .999), eps = 1e-8):
        self.learning_rate = lr
        self.named_params = dict(named_params)
        self.named_lr = named_lr
        self.beta1, self.beta2 = betas
        self.state = defaultdict(dict)
        self.epsilon = eps

    def step(self):
        for p1, p2 in self.named_params.items():
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
        for p1, p2 in self.named_params.items():
            if p2.grad is not None:
                if p2.grad.volatile:
                    p2.grad.data.zero_()
                else:
                    data = p2.grad.data
                    p2.grad = Variable(data.new().resize_as_(data).zero_())

class SGDSpec:
    def __init__(self, named_params, named_lr, lr, momentum = 0, weight_decay = 0):
        self.learning_rate = lr
        self.named_params = dict(named_params)
        self.named_lr = named_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = defaultdict(dict)

    def step(self):
        for p1, p2 in self.named_params.items():
            if p2.grad is None:
                continue
            grad = p2.grad.data

            #state = self.state[p2]
            state = self.state[p1]

            # State initialization
            if self.weight_decay != 0:
                grad.add_(self.weight_decay, p2.data)

            if self.momentum != 0:
                if 'mom_buffer' not in state:
                    state['mom_buffer'] = torch.zeros_like(p2.data).add_(grad)
                else:
                    state['mom_buffer'].mul_(self.momentum).add_(1 - self.momentum, grad)
                grad = state['mom_buffer']

            """
            if p1 in self.named_mom:
                U_mom = self.named_mom[p1]
                if len(state) == 0:
                    state['mom_buffer'] = grad.new().resize_as_(grad).zero_()
                    state['mom_buffer'].add_(grad)
                else:
                    state['mom_buffer'].mul_(U_mom).addcmul_(1 - U_mom, grad)
                grad = state['mom_buffer']
            else:
                if self.momentum != 0:
                    if len(state) == 0:
                        state['mom_buffer'] = grad.new().resize_as_(grad).zero_()
                        state['mom_buffer'].mul_(self.momentum).add_(grad)
                    else:
                        state['mom_buffer'].mul_(self.momentum).add_(1 - self.momentum, grad)
                    grad = state['mom_buffer']
            """

            if p1 in self.named_lr:
                wd = self.named_lr[p1]
                p2.data.addcmul_(-1, grad, wd)
            else:
                p2.data.add_(-self.learning_rate, grad)

    def zero_grad(self):
        for p1, p2 in self.named_params.items():
            if p2.grad is not None:
                if p2.grad.volatile:
                    p2.grad.data.zero_()
                else:
                    data = p2.grad.data
                    p2.grad = Variable(data.new().resize_as_(data).zero_())
