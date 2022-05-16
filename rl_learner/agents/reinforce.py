import numpy as np
import torch


class REINFORCE:
    def __init__(self, policy, lr=0.001, gamma=0.99):
        self.policy = policy
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = []

    @torch.no_grad()
    def act(self, s, training=True):
        self.policy.train(training)
        probs = self.policy(s)
        a = torch.multinomial(probs, 1) if training else torch.argmax(probs, dim=-1, keepdim=True)

        return a.cpu().numpy()

    def learn(self):
        self.policy.train()
        s, a, r, s_prime, done = map(lambda x: np.stack(x), zip(*self.buffer))

        ret = np.copy(r)
        for t in reversed(range(len(r) - 1)):
            ret[t] += self.gamma * ret[t+1]

        s, a, ret = map(lambda x: torch.as_tensor(x), [s, a, ret])
        a = a.unsqueeze(1)
        ret = ret.unsqueeze(1)

        probs = self.policy(s)
        log_probs = torch.log(probs.gather(1, a.long()))
        loss = -(log_probs * ret).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process(self, transition):
        # Process for each step
        self.buffer.append(transition)

        # Process for each episode
        if transition.done:
            self.learn()
            self.buffer = []
