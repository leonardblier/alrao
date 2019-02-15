r"""
These optimizers are copies of the original pytorch optimizers with one change:
    They take 2 specific arguments: named_parameters (instead of params) and named_lr.
    While looping over the dictionary named_parameters, these optimizers check whether
    the name of the current layer is in named_lr. If not, the optimization is
    exaclty the same as in the original optimizer. If it is the case, it looks
    into named_lr and take the tensor of specific learning rates associated to
    this layer, then uses it to update the layer.
"""

import math
import numpy as np
import torch
import torch.optim as optim
from .learningratesgen import lr_sampler_generic, generator_randomlr_neurons 


def init_alrao_optimizer(net, n_last_layers, minlr, maxlr, optim_name = 'SGD', momentum = 0., weight_decay = 0.):
    if n_last_layers > 1:
        last_layers_lr = [np.exp(
            np.log(minlr) + k / (n_last_layers - 1) * (np.log(maxlr) - np.log(minlr))
        ) for k in range(n_last_layers)]
        print(("Classifiers LR:" + n_last_layers * "{:.1e}, ").format(*tuple(last_layers_lr)))
    else:
        last_layers_lr = [minlr]

    lr_sampler = lr_sampler_generic(minlr, maxlr)
    lr_internal_nn = generator_randomlr_neurons(net.internal_nn, lr_sampler)

    if optim_name == 'SGD':
        optimizer = SGDAlrao(net.parameters_internal_nn(),
                             lr_internal_nn,
                             net.last_layers_parameters_list(),
                             last_layers_lr,
                             momentum = momentum,
                             weight_decay = weight_decay)
    elif optim_name == 'Adam':
        optimizer = AdamAlrao(net.parameters_internal_nn(),
                              lr_internal_nn,
                              net.last_layers_parameters_list(),
                              last_layers_lr)

    return optimizer

def alrao_step(net, optimizer, criterion, targets, catch_up = False, remove_non_numerical = True):
    """
    Perform an update step for a model learned with Alrao. The update is made assuming that
    only one forward pass has been performed since the last update.

    Arguments:
        net: model to learn
        optimizer: optimizer used
        criterion: loss used over each last layer of 'net'
        targets: the loss compares the stored output of each last layer of 'net' with 'targets'
        catch_up: set to True to activate the 'catch_up' mode of Alrao
        remove_non_numerical: if set to True, the gradients are not backpropagated from last 
            layers with infinite output
    """
    optimizer.last_layers_zero_grad()
    newx = net.last_x.detach()
    for last_layer in net.last_layers():
        loss_last_layer = criterion(last_layer(newx), targets)
        if torch.isfinite(loss_last_layer).all() or (not remove_non_numerical): 
            loss_last_layer.backward()

    optimizer.step()
    net.update_switch(targets, catch_up = catch_up)
    optimizer.update_posterior(net.posterior())

def sample_best_candidate(y):
    """
    Sample the most likely output among the outputs of an Alrao model.

    Arguments:
        y: tuple of 'outs' and 'ps':
            outs: tensor of size N * K
    """
    outs, ps = y
    _, i_max = ps.max(0)
    return outs[:, :, i_max]


class AdamSpec(optim.Optimizer):
    """
    Adam modification for the internal NN with Alrao

    Arguments:
        params: Iterator over the parameters
        lr_params: Iterator over the learning rates tensor. Same length than params, and
                    each lr tensor must have the same shape than the corresponding parameter
        other: Exactly the same as usual Adam
    """
    def __init__(self, params, lr_params, betas=(.9, .999), eps=1e-8,
                 weight_decay=0, amsgrad=False):

        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(AdamSpec, self).__init__(params, defaults)

        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]

    def __setstate__(self, state):
        super(AdamSpec, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """
        Compute a step of the optimizer
        """
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
                if not state:
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
    """
    SGD modification for the internal NN with Alrao

    Arguments:
        params: Iterator over the parameters
        lr_params: Iterator over the learning rates tensor. Same length than params, and
                    each lr tensor must have the same shape than the corresponding parameter
        other: Exactly the same as usual SGD
    """
    def __init__(self, params, lr_params, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGDSpec, self).__init__(params, defaults)

        for group in self.param_groups:
            group["lr_list"] = [lr for lr in lr_params]


    def __setstate__(self, state):
        super(SGDSpec, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """
        Compute a step of the optimizer
        """
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
    """
    TODO
    """
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
                    update = torch.matmul(lr_p, d_p.transpose(0, -2)).transpose(0, -2)
                    p.data.add_(-1, update)
                else:
                    raise ValueError


class SGDRandDirWeightsSpec(optim.Optimizer):
    """
    TODO
    """
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


class OptAlrao:
    """
    Generic class for Alrao's optimisation methods.
    """
    def __init__(self):
        pass

    def update_posterior(self, posterior):
        self.posterior = posterior

    def step(self):
        if self.opt_internal_nn is not None:
            self.opt_internal_nn.step()
        for opt_last_layer in self.opt_last_layers:
            opt_last_layer.step()

    def last_layers_zero_grad(self):
        for opt in self.opt_last_layers:
            opt.zero_grad()

    def zero_grad(self):
        if self.opt_internal_nn is not None:
            self.opt_internal_nn.zero_grad()
        for opt in self.opt_last_layers:
            opt.zero_grad()


class SGDAlrao(OptAlrao):
    """
    Alrao-SGD optimisation method.
    Arguments:
        parameters_internal_nn: iterator over the internal NN parameters
        lr_internal_nn: Iterator over the internal NN learning rates
        last_layer_parameters_list: List of iterators over each last_layer parameters
        last_layers_lr: List of iterators over each last layer learning rates
        other: as normal SGD
    """
    def __init__(self, parameters_internal_nn, lr_internal_nn, last_layers_parameters_list,
                 last_layers_lr, momentum=0., weight_decay=0.):

        super(SGDAlrao, self).__init__()
        self.opt_internal_nn = SGDSpec(parameters_internal_nn, lr_internal_nn,
                                        momentum=momentum, weight_decay=weight_decay)
        self.last_layers_lr = last_layers_lr
        self.opt_last_layers = \
            [optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay) \
             for parameters, lr in zip(last_layers_parameters_list, last_layers_lr)]


class AdamAlrao(OptAlrao):
    """
    Alrao-SGD optimisation method.
    Arguments:
        parameters_internal_nn: iterator over the internal NN parameters
        lr_internal_nn: Iterator over the internal NN learning rates
        last_layer_parameters_list: List of iterators over each last layer parameters
        last_layers_lr: List of iterators over each last layer learning rates
        other: as normal Adam
    """
    def __init__(self, parameters_internal_nn, lr_internal_nn, last_layers_parameters_list,
                 last_layers_lr, **kwargs):

        super(AdamAlrao, self).__init__()

        self.opt_internal_nn = AdamSpec(parameters_internal_nn, lr_internal_nn, **kwargs)
        self.opt_last_layers = [optim.Adam(parameters, lr, **kwargs) \
             for parameters, lr in zip(last_layers_parameters_list, last_layers_lr)]
