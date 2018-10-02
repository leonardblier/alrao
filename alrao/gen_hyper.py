import os
import random
import torch
import torch.nn as nn

# Learning rate
def fdet_lr_log(minLR, maxLR):
    M_minLR = minLR
    M_dstLR = maxLR - minLR

    def f(t):
        ret = t.new().resize_as_(t)
        if t.size()[0] == 1:
            ret[0].fill_(pow(10, M_minLR))
        else:
            for i in range(t.size()[0]):
                log_lr = M_minLR + M_dstLR * (i / (t.size()[0] - 1))
                ret[i].fill_(pow(10, log_lr))
        return ret

    return f

def fdet_lr_unif(lr):
    M_lr = lr

    def f(t):
        ret = t.new().resize_as_(t).fill_(M_lr)
        return ret

    return f

# frand_lr_log: TODO
def frand_lr_log(minLR, maxLR):
    M_minLR = minLR
    M_maxLR = maxLR

    def f(t):
        rd = random.uniform(M_minLR, M_maxLR)
        #includes.outputText('log10(lr) = ' + repr(rd + math.log10(args.lr)))
        ret = t.new().resize_as_(t).fill_(pow(10, rd))
        return ret

    return f

# Momentum TODO
def fdet_mom_id(t):
    ret = t.new().resize_as_(t)
    mom_dst = mom_max - mom_min
    for i in range(t.size()[0]):
        mom = mom_min + mom_max * (i / (t.size()[0] - 1))
        ret[i].fill_(mom)

    return ret

def fdet_mom_unif(t, u = 0):
    ret = t.new().resize_as_(t).fill_(u)
    return ret
