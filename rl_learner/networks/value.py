import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPStateValue(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=512):
        super(MLPStateValue, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def _format(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return x

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
