import os
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Switch(nn.Module):
    def __init__(self, nb_models, theta = .1):
        super(Switch, self).__init__()

        self.nb_models = nb_models
        self.theta = theta
        self.t = 1

        self.w_a = torch.FloatTensor(nb_models).fill_(theta * (1 / nb_models))
        self.w_b = torch.FloatTensor(nb_models).fill_((1 - theta) * (1 / nb_models))
        self.posterior = torch.FloatTensor(nb_models).fill_(1 / nb_models)
        self.nll = nn.NLLLoss()

    def Supdate(self, lst_x, y):
        lst_px = [None] * len(lst_x)
        for i in range(self.nb_models):
            lst_px[i] = math.exp(-self.nll(lst_x[i], y).data[0])

        if self.nb_models > 1:
            for i in range(self.nb_models):
                self.w_a[i] *= lst_px[i]
                self.w_b[i] *= lst_px[i]

            pool = (1 / self.t) * self.w_a.sum()
            self.w_a = self.w_a * (1 - 1 / self.t) + self.theta * pool / self.nb_models
            self.w_b = self.w_b + (1 - self.theta) * pool / self.nb_models

            U_norm = self.w_a.sum() + self.w_b.sum()
            self.w_a = self.w_a / U_norm
            self.w_b = self.w_b / U_norm
            self.posterior = self.w_a + self.w_b

            self.t += 1

    def forward(self, lst_x):
        if not self.training:
            _, i_max = torch.max(self.posterior, 0)
            return lst_x[i_max[0]]
        else:
            ret = Parameter(lst_x[0].data.new().resize_as_(lst_x[0].data).zero_(), requires_grad = False)
            for i in range(self.nb_models):
                ret += lst_x[i] * self.posterior[i]

            self.retain_ret = ret.data
            return ret
