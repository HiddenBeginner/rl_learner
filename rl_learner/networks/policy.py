import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDiscretePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(512, )):
        super(MLPDiscretePolicy, self).__init__()
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
        x = F.softmax(self.output_layer(x), dim=-1)

        return x


class MLPContinuousPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(512, )):
        super(MLPContinuousPolicy, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.mu_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], output_dim)

    def _format(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return x

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))

        return mu, log_std.exp()


class MLPDeterministicPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(512, )):
        super(MLPDeterministicPolicy, self).__init__()
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
        x = torch.tanh(self.output_layer(x))

        return x
