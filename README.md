The purpose of this repository is to bridge the gap between study-level implementations and practical package implementations in reinforcement learning. In particular, this repository aims to minimally modularize each component in RL while allowing users to run multiple experiments on various configurations. All codes are heavily based on the following:

- [@kakaoenterprise/JORLDY](https://github.com/kakaoenterprise/JORLDY)
- [@openai/spinningup](https://github.com/openai/spinningup)
- [@seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL)
- [@mimoralea/gdrl](https://github.com/mimoralea/gdrl)

<br>

## Agents
|Agent|Action type|reference |
|:---:|:---:|---:|
|[REINFORCE](https://github.com/HiddenBeginner/rl_learner/blob/master/rl_learner/agents/reinforce.py#L7)|Discrete/Continuous|Chapter 13.3, [1]|
|[REINFORCE with baseline](https://github.com/HiddenBeginner/rl_learner/blob/master/rl_learner/agents/reinforce.py#L66)|Discrete/Continuous|Chapter 13.4, [1]|
|[ActorCritic](https://github.com/HiddenBeginner/rl_learner/blob/master/rl_learner/agents/actor_critic.py#L7)|Discrete/Continuous|Chapter 13.5, [1]|

<br>

## References
[1] Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press.<br>
