import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

#TODO : Let choose the hyperparameters of the switch !
class Switch(nn.Module):
    """
    This class implements the switch distribution
    """
    def __init__(self, nb_models, theta = .9999, alpha=0.001):
        super(Switch, self).__init__()

        self.nb_models = nb_models
        self.theta = theta
        self.alpha = alpha
        self.t = 1

        self.register_buffer("w_a", torch.FloatTensor(nb_models).fill_(theta * (1 / nb_models)))
        self.register_buffer("w_b", torch.FloatTensor(nb_models).fill_((1 - theta) * (1 / nb_models)))
        self.register_buffer("posterior", torch.FloatTensor(nb_models).fill_(1 / nb_models))
        
    def piT(self, t):
        """
        Reprent the prior pi_t in algorithm 1
        """
        # Should keep former value
        return (1.-self.alpha)
    
    def Supdate(self, lst_x, y):
        """
        This implements algorithm 1 in "Catching Up Faster in Bayesian 
        Model Selection and Model Averaging
        """
        #px = torch.cat([(-F.nll_loss(x, y)).exp() for x in lst_x]).data
        px = torch.cat([-F.nll_loss(x, y) for x in lst_x]).data

        from math import isnan
        if any(isnan(p) for p in px):
            stop
        if self.nb_models > 1:
            self.w_a *= px
            self.w_b *= px

            pit = self.piT(self.t)
            pool = pit * self.w_a.sum()
            self.w_a = self.w_a * (1 - pit) + self.theta * pool / self.nb_models
            self.w_b = self.w_b + (1 - self.theta) * pool / self.nb_models

            U_norm = self.w_a.sum() + self.w_b.sum()
            self.w_a = self.w_a / U_norm
            self.w_b = self.w_b / U_norm
            self.posterior = self.w_a + self.w_b
                            
            self.t += 1

    def forward(self, lst_x):
        tensor_x = torch.stack(lst_x,-1)
        ret = tensor_x.matmul(Variable(self.posterior, requires_grad=False))
        self.retain_ret = ret.data
        return ret
