import numpy as np


def l2params(model):
    for name, p in model.named_parameters():
        #l += p.data.pow(2).sum()
        print("{} : \tL2Params : \t {:.5f} \t L2Grad : \t {:.5f}".format(
            name,
            float(p.data.pow(2).sum()) / np.prod(p.data.size()),
            float(p.grad.data.pow(2).sum()) / np.prod(p.data.size()),
        ))
              
    print("  ")

def l2grad(model):
    l = 0.
    print("L2 GRAD")
    for name, p in model.named_parameters():
        #l += p.data.pow(2).sum()
        print("{}:\t{:.3f}".format(name, p.grad.data.pow(2).sum()))
    print("  ")
    #return l
