import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class REINFORCE:
    def __init__(self, policy, action_type='discrete', lr=0.001, gamma=0.99):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = policy.to(self.device)
        self.action_type = action_type
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = []

    @torch.no_grad()
    def act(self, s, training=True):
        self.policy.train(training)

        s = torch.as_tensor(s).float().to(self.device)
        if self.action_type == 'discrete':
            probs = self.policy(s)
            a = torch.multinomial(probs, 1) if training else torch.argmax(probs, dim=-1, keepdim=True)

        else:
            mu, std = self.policy(s)
            z = torch.normal(mu, std) if training else mu
            a = torch.tanh(z)

        return a.cpu().numpy()

    def learn(self):
        self.policy.train()
        s, a, r, _, _ = map(lambda x: np.stack(x), zip(*self.buffer))

        ret = np.copy(r)
        for t in reversed(range(len(r) - 1)):
            ret[t] += self.gamma * ret[t + 1]

        s, a, ret = map(lambda x: torch.as_tensor(x).float().to(self.device), [s, a, ret])
        ret = ret.unsqueeze(1)

        if self.action_type == 'discrete':
            probs = self.policy(s)
            log_probs = torch.log(probs.gather(1, a.long()))

        else:
            mu, std = self.policy(s)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(a, -1.0 + 1e-7, 1.0 - 1e-7))
            log_probs = m.log_prob(z)

        loss = -(log_probs * ret).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process(self, transition):
        # Process for each step
        self.buffer.append(transition)

        # Process for each episode
        if transition[-1]:
            self.learn()
            self.buffer = []


class BaselineREINFORCE(REINFORCE):
    def __init__(
        self,
        policy,
        value,
        action_type='discrete',
        policy_lr=0.001,
        value_lr=0.001,
        gamma=0.99,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = policy.to(self.device)
        self.value = value.to(self.device)
        self.action_type = action_type
        self.gamma = gamma
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), value_lr)
        self.buffer = []

    def learn(self):
        self.policy.train()
        self.value.train()
        s, a, r, _, _ = map(lambda x: np.stack(x), zip(*self.buffer))

        ret = np.copy(r)
        for t in reversed(range(len(ret) - 1)):
            ret[t] += self.gamma * ret[t + 1]

        s, a, ret = map(lambda x: torch.as_tensor(x).float().to(self.device), [s, a, ret])
        ret = ret.unsqueeze(1)

        if self.action_type == 'discrete':
            probs = self.policy(s)
            log_probs = torch.log(probs.gather(1, a.long()))

        else:
            mu, std = self.policy(s)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(a, -1.0 + 1e-7, 1.0 - 1e-7))
            log_probs = m.log_prob(z)

        v = self.value(s)
        policy_loss = -(log_probs * (ret - v.detach())).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        value_loss = F.mse_loss(v, ret)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
