import random

# Learning rate
"""
This files implements four different ways to choose learning rates for a tensor.
Each of the following functions build a function which for each tensor t returns a tensor ret
which has the same size than t, and corresponds to its learning rates.

"""

def fdet_lr_log(minLR, maxLR):
    """
    Build a function which compute for each tensor t a tensor of learning rate deterministically,
    by spreading log-uniformly the learning rates over the interval (minLR, maxLR)
    Arguments:
        minLR, maxLR: Define the interval on which the learning rates are spread
    Returns:
        ret: a function which build a tensor of learning rate for each tensor t
    """
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
    """
    Build a function which compute for each tensor t a tensor of learning rate, by setting
    all the values to lr.
    Arguments:
        lr: The value of the learning rate
    Returns:
        ret: a function which build a tensor of learning rate for each tensor t
    """
    M_lr = lr

    def f(t):
        ret = t.new().resize_as_(t).fill_(M_lr)
        return ret

    return f

# frand_lr_log: TODO
def frand_lr_log(minLR, maxLR):
    """
    Build a function which compute for each tensor t a tensor of learning rate randomly
    with the log-uniform distribution over the interval (minLR, maxLR)
    Arguments:
        minLR, maxLR: Define the interval on which the learning rates are sampled.
    Returns:
        ret: a function which build a tensor of learning rate for each tensor t
    """
    M_minLR = minLR
    M_maxLR = maxLR

    def f(t):
        rd = random.uniform(M_minLR, M_maxLR)
        ret = t.new().resize_as_(t).fill_(pow(10, rd))
        return ret

    return f

# Momentum TODO
def fdet_mom_id(t, mom_min, mom_max):
    """
    TODO
    """
    ret = t.new().resize_as_(t)
    mom_dst = mom_max - mom_min
    for i in range(t.size()[0]):
        mom = mom_min + mom_max * (i / (t.size()[0] - 1))
        ret[i].fill_(mom)

    return ret

def fdet_mom_unif(t, u=0):
    ret = t.new().resize_as_(t).fill_(u)
    return ret
