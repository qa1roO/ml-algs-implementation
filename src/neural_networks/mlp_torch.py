import numpy as np
import torch.nn as nn

class MyTorchMLP(nn.Module):
    def __init__(self, n_features, n_hidden=128, activation="relu"):
        super().__init__()

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }

        if activation not in activations: raise ValueError("activation must be relu/tanh/sigmoid")

        act = activations[activation]

        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            act,
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.net(x)