import numpy as np
from torch.utils.data.dataset import Dataset


def l2params(model):
    """
    Print the l2 norms of each parameter tensor and of its gradient with its name
    """
    for name, p in model.named_parameters():
        print("{} : \tL2Params : \t {:.5f} \t L2Grad : \t {:.5f}".format(
            name,
            float(p.data.pow(2).sum()) / np.prod(p.data.size()),
            float(p.grad.data.pow(2).sum()) / np.prod(p.data.size()),
        ))

    print("  ")




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
