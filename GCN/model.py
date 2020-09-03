from math import ceil
from typing import Tuple

import torch
from scipy.stats import norm, binom_test
from torch import nn
from statsmodels.stats.proportion import proportion_confint

class GCN(nn.Module):
    """
    A simple graph convolutional neural network for Node Classification of Cora Dataset.
    """

    def __init__(self):
        super(GCN, self).__init__()
        self.sequential = nn.Sequential(
                             nn.Conv2d(1, 5, 5),
                             nn.ReLU(),
                             nn.BatchNorm2d(5),
                             nn.MaxPool2d(2),
                             nn.Conv2d(5, 5, 5),
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             nn.Flatten(),
                             nn.Linear(80, 10),
                            )

    def forward(self, input):
        assert input.min() >= 0 and input.max() <= 1.
        return self.sequential(input)

    def device(self):
        """
        Convenience function returning the device the model is located on.
        """
        return next(self.parameters()).device