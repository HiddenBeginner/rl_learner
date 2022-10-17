import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPStateValue(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dims=(512, )):
        super(MLPStateValue, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return x

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)

        return x


class MLPActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim=1, hidden_dims=(512, )):
        super(MLPActionValue, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return x

    def forward(self, s, a):
        s = self._format(s)
        a = self._format(a)
        x = torch.cat((s, a), axis=-1)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)

        return x
