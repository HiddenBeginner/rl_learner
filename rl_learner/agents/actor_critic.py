import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic:
    def __init__(
        self,
        policy,
        value,
        action_type='discrete',
        policy_lr=0.001,
        value_lr=0.001,
        gamma=0.99,
    ):
        self.action_type = action_type
        self.gamma = gamma
        self.policy = policy
        self.value = value
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), value_lr)
        self.buffer = []

    @torch.no_grad()
    def act(self, s, training=True):
        self.policy.train(training)
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
        self.value.train()
        s, a, r, s_prime, done = map(lambda x: np.stack(x), zip(*self.buffer))
        s, a, r, s_prime, done = map(lambda x: torch.as_tensor(x).float(), [s, a, r, s_prime, done])
        r = r.unsqueeze(1)
        done = done.unsqueeze(1)

        if self.action_type == 'discrete':
            probs = self.policy(s)
            log_probs = torch.log(probs.gather(1, a.long()))

        else:
            mu, std = self.policy(s)
            m = Normal(mu, std)
            z = torch.atanh(torch.clip(a, -1.0 + 1e-7, 1.0 - 1e-7))
            log_probs = m.log_prob(z)

        td_target = r + (1 - done) * self.gamma * self.value(s_prime)
        td_error = td_target - self.value(s)

        value_loss = F.mse_loss(self.value(s), td_target.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        policy_loss = - (log_probs * td_error.detach()).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def process(self, transition):
        self.buffer.append(transition)

        if transition.done:
            self.learn()
            self.buffer = []
