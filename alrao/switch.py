"""
Switch model averaging
"""


import math
from numbers import Number
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO : Let choose the hyperparameters of the switch !
class Switch(nn.Module):
    """
    Model averaging method 'switch'.
    See van Erven and Grünwald (2008):
     (A) https://arxiv.org/abs/0807.1005,
     (B) http://papers.nips.cc/paper/3277-catching-up-faster-in-bayesian-model-selection-and-model-averaging.pdf.

    This class manages model averaging and updates its parameters using the update
        rule given in algorithm 1 of (B).

    Parameters:
        loss: loss used in the model (subclass of pytorch's '_Loss')
            loss(output, target, size_average = False) returns the loss embedded into a 0-dim tensor
            the option 'size_average = True' returns the averaged loss
    """
    def __init__(self, nb_models, theta = .9999, alpha = .001, save_ll_perf = False, 
            task = 'classification', loss = None):
        super(Switch, self).__init__()

        self.task = task
        if task == 'classification' and loss is None:
            self.loss = F.nll_loss
        elif loss is not None:
            self.loss = loss
        else:
            raise ValueError('Invalid combination task/loss: {} / {}.'.format(task, loss))
        self.nb_models = nb_models
        self.theta = theta
        self.alpha = alpha
        self.t = 1

        self.save_ll_perf = save_ll_perf

        self.register_buffer("logw", torch.zeros((2, nb_models), requires_grad = False))
        self.register_buffer("logposterior",
                             torch.full((nb_models,),
                             -np.log(nb_models), requires_grad = False))
        self.logw[0].fill_(np.log(theta))
        self.logw[1].fill_(np.log(1 - theta))
        self.logw -= np.log(nb_models)

        if self.save_ll_perf:
            self.reset_ll_perf()

    def reset_ll_perf(self):
        """
        Resets the performance record of last layers models
        """
        self.ll_loss = [0 for _ in range(self.nb_models)]
        self.ll_correct = [0 for _ in range(self.nb_models)]
        self.ll_total = 0

    def get_ll_perf(self):
        """
        Return the performance (loss and acc) of each last layer
        """
        if self.task == 'classification':
            return [(loss / self.ll_total, corr / self.ll_total) \
                    for (loss, corr) in zip(self.ll_loss, self.ll_correct)]
        elif self.task == 'regression':
            return [loss / self.ll_total \
                    for loss in self.ll_loss]

    def piT(self, t):
        """
        Prior  pi_T in algorithm 1 of (B).
        """
        return 1 / (t + 1) #(1.-self.alpha)

    def Supdate(self, lst_ll_out, y):
        """
        Switch update rule given in algorithm 1 of (B).

        Arguments:
            lst_ll_out: list of the outputs of the models, which are supposed to be
                tensors of log-probabilities
            y: tensor of targets
        """
        if self.save_ll_perf:
            self.ll_total += 1
            for (k, x) in enumerate(lst_ll_out):
                self.ll_loss[k] += self.loss(x, y).item()
                if self.task == 'classification':
                    self.ll_correct[k] += (torch.max(x, 1)[1]).eq(y.data).sum().item() / y.size(0)

        # px is the tensor of the log probabilities of the mini-batch for each last layer
        #for x in lst_logpx:
        #    print(self.loss(x, y))
        logpx = torch.stack([-self.loss(x, y) for x in lst_ll_out], dim = 0).detach()

        if any(math.isnan(p) for p in logpx):
            raise ValueError
        if self.nb_models == 1:
            return

        self.logw += logpx
        pit = self.piT(self.t)
        logpool = log_sum_exp(self.logw[0]) + np.log(pit)
        self.logw[0] += np.log(1 - pit)

        addtensor = torch.zeros_like(self.logw)
        addtensor[0].fill_(np.log(self.theta))
        addtensor[1].fill_(np.log(1-self.theta))

        self.logw = log_sum_exp(torch.stack([self.logw,
            addtensor + logpool - np.log(self.nb_models)], dim = 0), dim = 0)

        self.logw -= log_sum_exp(self.logw)
        self.logposterior = log_sum_exp(self.logw, dim = 0)
        self.t += 1

    def forward(self, lst_ll_out):
        """
        Computes the average of the outputs of the different models.

        Arguments:
            lst_ll_out: list of the outputs of the models, which are supposed to be
                tensors of log-probabilities
        """
        if self.task == 'classification':
            return log_sum_exp(torch.stack(lst_ll_out, -1) + self.logposterior, dim = -1)
        elif self.task == 'regression':
            return torch.stack(lst_ll_out, -1), self.logposterior.exp()


def log_sum_exp(tensor, dim=None):
    """
    Numerically stable implementation of the operation.

    tensor.exp().sum(dim, keepdim).log()
    From https://github.com/pytorch/pytorch/issues/2591
    """
    if dim is not None:
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        tensor0 = tensor - m
        return m.squeeze(dim=dim) + torch.log(torch.sum(torch.exp(tensor0), dim=dim))
    else:
        m = torch.max(tensor)
        sum_exp = torch.sum(torch.exp(tensor - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
