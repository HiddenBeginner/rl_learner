import gym
import numpy as np


class ScalarActionToArrayEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        if np.ndim(action) > 0:
            return self.env.step(action[0])
        else:
            return self.env.step(action)


def make(env_name):
    env = gym.make(env_name)

    if len(env.action_space.shape) == 0:
        env = ScalarActionToArrayEnv(env)

    return env
