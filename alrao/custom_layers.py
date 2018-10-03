"""
Custom layers for Alrao
"""

import torch.nn as nn
import torch.nn.functional as F



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
