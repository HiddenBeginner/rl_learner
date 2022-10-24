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
|[DDPG](https://github.com/HiddenBeginner/rl_learner/blob/master/rl_learner/agents/ddpg.py#L9)|Continuous|Lillicrap et al., 2016 [2]|
|[TD3](https://github.com/HiddenBeginner/rl_learner/blob/master/rl_learner/agents/td3.py#L9)|Continuous|Fujimoto et al., 2018 [3]|

<br>

## References
[1] Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press.<br>
[2] Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. “Continuous control with deep reinforcement learning.” In ICLR (Poster), 2016. http://arxiv.org/abs/1509.02971.<br>
[3] Fujimoto, Scott, Herke van Hoof, David Meger. “Addressing Function Approximation Error in Actor-Critic Methods”. In Proceedings of the 35th International Conference on Machine Learning,  Jennifer Dy, Andreas Krause, 80:1587–96. Proceedings of Machine Learning Research. PMLR, 2018. https://proceedings.mlr.press/v80/fujimoto18a.html.

