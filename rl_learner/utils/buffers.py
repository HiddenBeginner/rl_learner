import numpy as np
import torch


class ReplayBuffer:
    """
    References
    ----------
    https://github.com/sfujim/TD3/blob/master/utils.py
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((max_size, state_dim))
        self.a = np.zeros((max_size, action_dim))
        self.r = np.zeros((max_size, 1))
        self.s_prime = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def __len__(self):
        return self.size

    def store(self, s, a, r, s_prime, done):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=100):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.done[ind]),
        )
