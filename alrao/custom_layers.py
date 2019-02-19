"""
Custom layers for Alrao
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .switch import log_sum_exp 


class LinearClassifier(nn.Module):
    """
    Linear classifier layer: a linear layer followed by a log_softmax activation
    """
    def __init__(self, in_features, n_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        """
        Forward pass method
        """
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class LinearClassifierRNN(nn.Module):
    """
    Linear classifier layer for RNNs: a decoder (linear layer) followed by a log_softmax activation
    """
    def __init__(self, nhid, ntoken):
        super(LinearClassifierRNN, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.ntoken = ntoken

    def init_weights(self):
        """
        Initialization
        """
        initrange = .1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, output):
        """
        Forward pass method: flattening of the output (specific to RNNs), which is processed by a
            linear layer followed by a log_softmax activation
        """
        ret = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return F.log_softmax(ret, dim=1)

class LinearRegressor(nn.Module):
    """
    Linear final layer of a regressor
    """
    def __init__(self, dim_input, dim_output):
        super(LinearRegressor, self).__init__()
        self.layer = nn.Linear(dim_input, dim_output)

    def forward(self, output):
        y = self.layer(output)
        if not torch.isfinite(y).all():
            raise ValueError
        return y

# Regression loss
# base loss class
class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

# loss adapted to one final layer
class L2LossLog(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', sigma2 = 1.):
        super(L2LossLog, self).__init__(size_average, reduce, reduction)
        self.sigma2 = sigma2

    def forward(self, input, target):
        if not torch.isfinite(input).all():
            raise ValueError

        return ((input - target).pow(2).sum(1) / (2 * self.sigma2)).mean() + \
                .5 * math.log(2 * math.pi * self.sigma2)

# loss adapted to the output of the switch
class L2LossAdditional(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', sigma2 = 1.):
        super(L2LossAdditional, self).__init__(size_average, reduce, reduction)
        self.sigma2 = sigma2

    def forward(self, input, target):
        means, ps = input
        if not torch.isfinite(means).all():
            raise ValueError

        # means: batch_size * out_size * nb_last_layers
        means = means.transpose(2, 1).transpose(1, 0)
        # means: nb_last_layers * batch_size * out_size
        log_probas_per_cl = -(means - target).pow(2).sum(2) / (2 * self.sigma2) - \
                .5 * math.log(2 * math.pi * self.sigma2)
        # log_probas_per_cl: nb_classifiers * batch_size
        log_probas_per_cl = log_probas_per_cl.transpose(0, 1)
        # log_probas_per_cl: batch_size * nb_classifiers
        log_probas_per_cl = log_probas_per_cl + ps.log()
        log_probas = log_sum_exp(log_probas_per_cl, dim = 1)
        return -log_probas.mean()
