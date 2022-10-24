from copy import deepcopy

import torch
import torch.nn.functional as F

from ..utils.buffers import ReplayBuffer


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor,
        critic1,
        critic2,
        actor_lr=0.0003,
        critic_lr=0.0003,
        tau=0.005,
        gamma=0.99,
        delay_freq=2,
        expl_noise=0.1,
        smoothing_noise=0.2,
        noise_clip=0.5,
        batch_size=100,
        warmup_steps=1000,
        buffer_size=int(1e6),
    ):
        self.total_it = 0
        self.tau = tau
        self.gamma = gamma
        self.delay_freq = delay_freq
        self.expl_noise = expl_noise
        self.smoothing_noise = smoothing_noise
        self.noise_clip = noise_clip

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)

        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic1 = deepcopy(self.critic1).to(self.device)
        self.target_critic2 = deepcopy(self.critic2).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

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

        with torch.no_grad():
            noise = (
                torch.rand_like(a) * self.smoothing_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            a_prime = (
                self.target_actor(s_prime) + noise
            ).clamp(-1.0, 1.0)

            Q_target = torch.min(self.target_critic1(s_prime, a_prime), self.target_critic2(s_prime, a_prime))
            td_target = r + (1 - done) * self.gamma * Q_target

        critic1_loss = F.mse_loss(self.critic1(s, a), td_target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_loss = F.mse_loss(self.critic2(s, a), td_target)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.delay_freq == 0:
            actor_loss = - self.critic1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for params, target_params in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

            for params, target_params in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

            for params, target_params in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

    def process(self, transition):
        self.total_it += 1
        self.buffer.store(*transition)
        if len(self.buffer) >= self.warmup_steps:
            self.learn()
