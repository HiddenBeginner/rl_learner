from copy import deepcopy

import torch
import torch.nn.functional as F

from ..utils.buffers import ReplayBuffer


class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor,
        critic,
        actor_lr=0.0003,
        critic_lr=0.0003,
        tau=0.005,
        gamma=0.999,
        expl_noise=0.2,
        batch_size=64,
        warmup_steps=1000,
        buffer_size=int(1e6),
    ):
        self.tau = tau
        self.gamma = gamma
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_actor = deepcopy(actor).to(self.device)
        self.target_critic = deepcopy(critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, s, training=True):
        self.actor.train(training)
        s = torch.as_tensor(s).float().to(self.device)
        a = self.actor(s)
        if training:
            a += torch.normal(0, self.expl_noise, size=a.size()).to(self.device)

        return torch.clip(a, -1.0, 1.0).cpu().numpy()

    def learn(self):
        s, a, r, s_prime, done = self.buffer.sample(self.batch_size)
        s, a, r, s_prime, done = map(lambda x: x.to(self.device), [s, a, r, s_prime, done])

        td_target = r + (1 - done) * self.gamma * self.target_critic(s_prime, self.target_actor(s_prime))
        critic_loss = F.mse_loss(self.critic(s, a), td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for params, target_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

        for params, target_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

    def process(self, transition):
        self.buffer.store(*transition)
        if len(self.buffer) >= self.warmup_steps:
            self.learn()
