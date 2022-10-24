import gym
import numpy as np


class DenormalizeContinuousAction(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        action = (0.5 * (action + 1)) * (
            self.env.action_space.high - self.env.action_space.low
        ) + self.env.action_space.low

        return self.env.step(action)


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

    if isinstance(env.action_space, gym.spaces.Box):
        env = DenormalizeContinuousAction(env)

    if len(env.action_space.shape) == 0:
        env = ScalarActionToArrayEnv(env)

    return env
