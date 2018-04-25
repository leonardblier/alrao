import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from numbers import Number
import pdb

#TODO : Let choose the hyperparameters of the switch !
class Switch(nn.Module):
    """
    This class implements the switch distribution
    """
    def __init__(self, nb_models, theta = .9999, alpha=0.001, save_cl_perf=False):
        super(Switch, self).__init__()

        self.nb_models = nb_models
        self.theta = theta
        self.alpha = alpha
        self.t = 1

        self.save_cl_perf = save_cl_perf
        
        #self.register_buffer("logw_a", torch.FloatTensor(nb_models))
        #self.register_buffer("logw_b", torch.FloatTensor(nb_models))

        self.register_buffer("logw", torch.zeros((2, nb_models)))
        
        self.register_buffer("logposterior", torch.FloatTensor(nb_models))

        self.logw[0].fill_(np.log(theta * (1 / nb_models)))
        self.logw[1].fill_(np.log((1 - theta) * (1 / nb_models)))
        
        self.logposterior.fill_(- np.log(nb_models))
        if self.save_cl_perf:
            self.reset_cl_perf()
            
    def reset_cl_perf(self):
        self.cl_loss = [0 for _ in range(self.nb_models)]
        self.cl_correct = [0 for _ in range(self.nb_models)]
        self.cl_total = 0

    def get_cl_perf(self):
        return [(loss / self.cl_total, corr / self.cl_total) \
                for (loss, corr) in zip(self.cl_loss, self.cl_correct)]
        
    def piT(self, t):
        """
        Reprent the prior pi_t in algorithm 1
        """
        return (1.-self.alpha)
    
    def Supdate(self, lst_x, y):
        """
        This implements algorithm 1 in "Catching Up Faster in Bayesian 
        Model Selection and Model Averaging
        """
        if self.save_cl_perf:
            self.cl_total += 1
            for (k, x) in enumerate(lst_x):
                self.cl_loss[k] += nn.NLLLoss()(x, y).data[0]
                self.cl_correct[k] += (torch.min(x.data, 1)[1]).eq(y.data).sum() / y.size()[0]
                
        # px is the tensor of the log probabilities of the mini-batch for each classifier
        px = torch.cat([-F.nll_loss(x.log(), y) for x in lst_x]).data

        from math import isnan
        if any(isnan(p) for p in px):
            stop
        if self.nb_models == 1:
            return None

        self.logw += px
        pit = self.piT(self.t)
        logpool = np.log(pit) + log_sum_exp(self.logw[0])
        self.logw[0] += np.log(1 - pit)

        addtensor = torch.zeros_like(self.logw).fill_(self.theta)
        addtensor[1].fill_(1-self.theta)

        self.logw = log_sum_exp(torch.stack([self.logw,
            addtensor.log() + logpool - np.log(self.nb_models)], dim=0), dim = 0)

        self.logw -= log_sum_exp(self.logw)
        self.logposterior = log_sum_exp(self.logw, dim=0)
        self.t += 1

    def forward(self, lst_x):
        tensor_x = torch.stack(lst_x,-1)
        ret = tensor_x.matmul(Variable(self.logposterior.exp(), requires_grad=False))
        self.retain_ret = ret.data
        return ret


    

def log_sum_exp(tensor, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    tensor.exp().sum(dim, keepdim).log()
    From https://github.com/pytorch/pytorch/issues/2591
    """
    if dim is not None:
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        tensor0 = tensor - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(tensor0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(tensor)
        sum_exp = torch.sum(torch.exp(tensor - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
