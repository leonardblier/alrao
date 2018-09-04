import numpy as np
from torch.utils.data.dataset import Dataset

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


#### TO BE REMOVED
class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
