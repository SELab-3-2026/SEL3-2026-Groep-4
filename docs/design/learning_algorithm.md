# Reinforcement Learning Algorithm

To control the *continuous* action space (the joints of the robot) based on sensor data, we require a reliable
Reinforcement Learning (RL) algorithm or optimization strategy.

We have chosen **Proximal Policy Optimization (PPO)** (Schulman et al., 2017).

## Rationale

PPO is an on-policy algorithm known for its stability and robustness (safe training without excessive variance). More
importantly, it requires relatively little hyperparameter tuning compared to other algorithms. Since NerveNet was
successfully trained using PPO (Wang et al., 2018), selecting PPO significantly reduces the risk of convergence
failures.

## Limitations and alternatives

Alternative learning algorithms include:

- **Twin Delayed DDPG (Fujimoto et al., 2018)**: TD3 is a strong off-policy alternative used in the SMP paper (Huang et
al., 2020). It is highly sample-efficient and reportedly excels at zero-shot adaptations. However, this approach would be
more complex and error-prone than with PPO.
- **Evolution strategies (ES)**: Evolution strategies are useful for optimizing Central Pattern Generators (CPGs), e.g.
CMA-ES, OpenAI-ES. While this method is easier to distribute and parallelize, ES typically scales worse with
exceptionally large observation spaces compared to gradient-based RL methods like PPO.

**References**

- Fujimoto, Scott, Herke Hoof, and David Meger. ‘Addressing Function Approximation Error in Actor-Critic Methods’. Proceedings of the 35th International Conference on Machine Learning, 3 July 2018, 1587-96. https://proceedings.mlr.press/v80/fujimoto18a.html.
- Huang, Wenlong, Igor Mordatch, and Deepak Pathak. ‘One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control’. arXiv:2007.04976. Preprint, arXiv, 9 July 2020. https://doi.org/10.48550/arXiv.2007.04976.
- Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. ‘Proximal Policy Optimization Algorithms’. arXiv:1707.06347. Preprint, arXiv, 28 August 2017. https://doi.org/10.48550/arXiv.1707.06347.
