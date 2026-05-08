import datetime
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

from experiment_logger import get_logger

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.dataclasses import EpisodeStatistics
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.MLPs import (
    Actor,
    AgentParams,
    GenericDenseLayersWithActivation,
    MessagePasser,
    OneDenseLayerMLP,
    Storage,
    build_adjacency,
)
from brittle_star_project.ppo import PPO
from brittle_star_project.environment import MorphMode
from brittle_star_project.utils import logged_jit


@logged_jit
def _clip_action(action: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(action, low, high)


def _compute_explained_variance(values: jnp.ndarray, returns: jnp.ndarray) -> float:
    var_returns = jnp.var(returns)
    explained_var = 1.0 - jnp.var(returns - values) / (var_returns + 1e-8)
    return float(explained_var)


@logged_jit
def _linear_schedule(count, minibatch_count, update_epochs, num_iterations, learning_rate):
    frac = 1.0 - (count // (minibatch_count * update_epochs)) / num_iterations
    return learning_rate * frac


def _get_action_and_value_noise(
    sensor: nn.Module,
    feature_extractor: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    message_passer: Optional[nn.Module],
    agent_state: TrainState,
    next_obs: jnp.ndarray,
    key,
    action_low,
    action_high,
):
    # (B, n_nodes, feat)
    hidden = apply_per_node(sensor, agent_state.params["sensor_params"], next_obs)

    if message_passer is not None:
        params = agent_state.params["message_passer_params"]
        # (n_nodes, feat) --> let each node talk with its neighbours ==> vmap over B dimension
        hidden = jax.vmap(lambda x: message_passer.apply(params, x))(hidden)

    hidden_critic = apply_shared(
        feature_extractor, agent_state.params["feature_extractor_params"], next_obs
    )

    mean, log_std = apply_per_node(actor, agent_state.params["actor_params"], hidden)
    log_std = jnp.clip(log_std, -5, 2)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=mean.shape)
    std = jnp.exp(log_std)

    raw_action = mean + noise * std
    flat_action = raw_action.reshape(
        raw_action.shape[0], -1
    )  # concat the per agent, keep the envs dim (batch, agent * action)
    flat_clipped_action = _clip_action(flat_action, action_low, action_high)

    logprob = -0.5 * (((raw_action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(
        axis=(-2, -1)
    )
    value = apply_shared(critic, agent_state.params["critic_params"], hidden_critic)

    return flat_clipped_action, raw_action, logprob, value.squeeze(-1), mean, std, key


def _step_once(
    carry,
    _,
    env_step_fn,
    num_envs: int,
    sensor: nn.Module,
    feature_extractor: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    message_passer: Optional[nn.Module],
    action_low,
    action_high,
):
    agent_state, episode_stats, obs, done, key, env_state, terminated_any, truncated_any = carry
    flat_clipped_action, raw_action, logprob, value, mean, std, key = _get_action_and_value_noise(
        sensor,
        feature_extractor,
        actor,
        critic,
        message_passer,
        agent_state,
        obs,
        key,
        action_low,
        action_high,
    )
    logger = get_logger()

    logger.debug(f"[_step_once] raw_action: {raw_action.shape}")
    logger.debug(f"[_step_once] clipped_action: {flat_clipped_action.shape}")

    # Supporting signals (often where mismatch originates)
    logger.debug(f"[_step_once] logprob: {logprob.shape}")
    logger.debug(f"[_step_once] value: {value.shape}")
    logger.debug(f"[_step_once] mean: {mean.shape}")
    logger.debug(f"[_step_once] std: {std.shape}")

    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, num_envs)

    # ---- ENV STEP ----
    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, num_envs)

    episode_stats, env_state, (next_obs, reward, next_done, terminated, truncated) = env_step_fn(
        episode_stats,
        env_state,
        flat_clipped_action,
        reset_rngs,
    )

    terminated_any = terminated_any | terminated
    truncated_any = truncated_any | truncated

    logger.debug(f"[_step_once] next_obs: {next_obs.shape}")
    logger.debug(f"[_step_once] reward: {reward.shape}")
    logger.debug(f"[_step_once] next_done: {next_done.shape}")

    storage = Storage(
        obs=obs,
        actions=raw_action,
        raw_actions=raw_action,
        logprobs=logprob,
        dones=done,
        values=value,
        rewards=reward,
        means=mean,
        stds=std,
        returns=jnp.zeros_like(reward),
        advantages=jnp.zeros_like(reward),
    )
    return (
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        key,
        env_state,
        terminated_any,
        truncated_any,
    ), storage


def _reward_fn(env_state, next_env_state):
    # if delta distance positive ==> brittle star walking away from target
    delta_distance = (
        next_env_state.observations["xy_distance_to_target"]
        - env_state.observations["xy_distance_to_target"]
    ).squeeze(-1)

    env_reward = next_env_state.reward
    clipped_env_reward = jnp.clip(100 * env_reward, -10, 10)

    time_penalty = 0.1
    distance_penalty = jnp.clip(0.5 * delta_distance, -0.5, 0.5)
    penalty = time_penalty + distance_penalty

    return jnp.where(next_env_state.terminated, 50.0, clipped_env_reward - penalty)


def _step_env_wrapped(
    episode_stats,
    env_state,
    action,
    reset_rngs,
    env_step_fn,
    reset_single_fn,
    obs_processor,
):
    next_env_state_pre_reset = env_step_fn(env_state, action)

    reward = _reward_fn(env_state, next_env_state_pre_reset)
    terminated = next_env_state_pre_reset.terminated
    truncated = next_env_state_pre_reset.truncated
    done = terminated | truncated

    new_episode_return = episode_stats.episode_returns + reward
    new_episode_length = episode_stats.episode_lengths + 1

    episode_stats = episode_stats.replace(
        episode_returns=new_episode_return * (1 - done),
        episode_lengths=new_episode_length * (1 - done),
        returned_episode_returns=jnp.where(
            done, new_episode_return, episode_stats.returned_episode_returns
        ),
        returned_episode_lengths=jnp.where(
            done, new_episode_length, episode_stats.returned_episode_lengths
        ),
    )

    def _maybe_reset(state_i, rng_i, do_reset_i):
        def _do(_):
            reset_state = reset_single_fn(rng=rng_i)

            def _cast_leaf(new_leaf, like_leaf):
                if like_leaf is None or new_leaf is None:
                    return new_leaf

                # Use jnp.asarray(...) to robustly get dtype for both JAX arrays and Python scalars.
                like_dtype = jnp.asarray(like_leaf).dtype

                # Avoid unnecessary work when already matching.
                if hasattr(new_leaf, "dtype") and new_leaf.dtype == like_dtype:
                    return new_leaf

                return jnp.asarray(new_leaf, dtype=like_dtype)

            # `lax.cond` requires both branches to return identical PyTree types/dtypes.
            return jax.tree_util.tree_map(_cast_leaf, reset_state, state_i)

        def _dont(_):
            return state_i

        return jax.lax.cond(do_reset_i, _do, _dont, operand=None)

    # Auto-reset done envs so rollouts continue with fresh episode initial states.
    next_env_state = jax.vmap(_maybe_reset)(next_env_state_pre_reset, reset_rngs, done)

    return (
        episode_stats,
        next_env_state,
        (obs_processor(next_env_state.observations), reward, done, terminated, truncated),
    )


def apply_per_node(net, params, x):
    # params: (nodes, ...)
    # x: (batch, nodes, feat)

    def apply_single_node(p, x_node):
        # x_node: (batch, feat)
        return jax.vmap(lambda xi: net.apply(p, xi))(x_node)

    return jax.vmap(apply_single_node, in_axes=(0, 1), out_axes=1)(params, x)


def apply_shared(net, params, x):
    # x: (batch, nodes, feat)
    # If the critic expects a single vector per environment:
    batch_size = x.shape[0]
    x_flattened = x.reshape(batch_size, -1)
    return jax.vmap(lambda xi: net.apply(params, xi))(x_flattened)


def _rollout_jit(
    agent_state,
    episode_stats,
    env_state,
    next_obs,
    next_done,
    key,
    max_steps,
    step_env_fn,
    num_envs: int,
    sensor: nn.Module,
    feature_extractor: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    message_passer: Optional[nn.Module],
    action_low,
    action_high,
):
    terminated_any0 = jnp.zeros((num_envs,), dtype=jnp.bool_)
    truncated_any0 = jnp.zeros((num_envs,), dtype=jnp.bool_)

    (
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            key,
            env_state,
            terminated_any,
            truncated_any,
        ),
        storage,
    ) = jax.lax.scan(
        partial(
            _step_once,
            sensor=sensor,
            feature_extractor=feature_extractor,
            actor=actor,
            critic=critic,
            message_passer=message_passer,
            env_step_fn=step_env_fn,
            num_envs=num_envs,
            action_low=action_low,
            action_high=action_high,
        ),
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            key,
            env_state,
            terminated_any0,
            truncated_any0,
        ),
        (),
        max_steps,
    )
    return (
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        storage,
        key,
        env_state,
        terminated_any,
        truncated_any,
    )


def _compute_gae_once(carry, inp, gamma, gae_lambda):
    advantages = carry
    nextdone, nextvalues, curvalues, reward = inp
    nextnonterminal = 1.0 - nextdone
    delta = reward + gamma * nextvalues * nextnonterminal - curvalues
    advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
    return advantages, advantages


def _compute_gae_jit(
    agent_state,
    storage,
    next_obs,
    next_done,
    gamma,
    gae_lambda,
    num_envs,
    feature_extractor,
    critic,
):
    next_value = apply_shared(
        critic,
        agent_state.params["critic_params"],
        apply_shared(feature_extractor, agent_state.params["feature_extractor_params"], next_obs),
    ).squeeze(-1)

    advantages = jnp.zeros((num_envs,))
    dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
    values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
    _, advantages = jax.lax.scan(
        partial(_compute_gae_once, gamma=gamma, gae_lambda=gae_lambda),
        advantages,
        (dones[1:], values[1:], values[:-1], storage.rewards),
        reverse=True,
    )
    returns = advantages + storage.values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return storage.replace(advantages=advantages, returns=returns)


@dataclass
class TrainingMeasurements:
    loss: jnp.ndarray
    pg_loss: jnp.ndarray
    v_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    approx_kl: jnp.ndarray
    avg_episodic_return: float
    explained_variance: float
    num_terminated: int
    num_truncated: int
    avg_terminated_length: Any
    avg_truncated_length: Any


class PPOTrainer:
    def __init__(
        self,
        cfg: BrittleStarConfig,
        env: BrittleStarJaxEnvWrapper,
        run_dir: str,
        run_name: str,
    ):
        self.cfg = cfg
        self.ppo = cfg.ppo
        self.experiment = cfg.experiment
        self.logging_cfg = cfg.logging
        self.env = env
        self.run_dir = run_dir
        self.run_name = run_name
        self.logger = get_logger()

        # Derived runtime fields
        self.batch_size = self.ppo.num_envs * self.ppo.num_steps
        self.num_iterations = self.ppo.total_timesteps // self.batch_size

        self.key = jax.random.PRNGKey(self.experiment.seed)

        self.morph_mode = self.cfg.morphology.morph_mode

        self.segments_per_arm = jnp.asarray(self.cfg.morphology.segments_per_arm, dtype=jnp.int32)
        self.num_segments = self.segments_per_arm.sum().item()
        self.num_arms = jnp.where(self.segments_per_arm > 0, 1, 0).sum().item()

        self.logger.info(f"[INIT]: Used morphology mode {self.morph_mode}")
        self.adj = build_adjacency(cfg.morphology.segments_per_arm, self.morph_mode)

        (
            self.sensor,
            self.message_passer,
            self.actor,
            self.feature_extractor,
            self.critic,
            self.needed_copies,
            self.agent_indices,
        ) = self._init_agent()

        self.sensor.apply = logged_jit(self.sensor.apply)
        self.feature_extractor.apply = logged_jit(self.feature_extractor.apply)
        self.actor.apply = logged_jit(self.actor.apply)
        self.critic.apply = logged_jit(self.critic.apply)

        # Build the centralized observation processor: derive -> normalize -> pad -> flatten.
        self.obs_processor = create_obs_processor(
            bounds_dict=self.cfg.obs_bounds.to_bounds_dict(),
            needed_copies=self.needed_copies,
            num_arms=self.num_arms,
            morph_mode=self.morph_mode,
            padding_masks=self.env.padding_masks,
            segments_per_arm=self.segments_per_arm,
            agent_indices=self.agent_indices,
        )

        action_low = jnp.asarray(self.env.single_action_space.low, dtype=jnp.float32)
        action_high = jnp.asarray(self.env.single_action_space.high, dtype=jnp.float32)

        self._rollout_jit = logged_jit(
            partial(
                _rollout_jit,
                max_steps=self.ppo.num_steps,
                step_env_fn=partial(
                    _step_env_wrapped,
                    env_step_fn=self.env.step,
                    reset_single_fn=self.env.raw.reset,
                    obs_processor=self.obs_processor,
                ),
                num_envs=self.ppo.num_envs,
                sensor=self.sensor,
                feature_extractor=self.feature_extractor,
                actor=self.actor,
                critic=self.critic,
                message_passer=self.message_passer,
                action_low=action_low,
                action_high=action_high,
            )
        )
        self._compute_gae_jit = logged_jit(
            partial(
                _compute_gae_jit,
                num_envs=self.ppo.num_envs,
                gamma=self.ppo.gamma,
                gae_lambda=self.ppo.gae_lambda,
                feature_extractor=self.feature_extractor,
                critic=self.critic,
            )
        )

        def apply_sensor(p, x):
            return apply_per_node(self.sensor, p, x)

        def apply_actor(p, x):
            return apply_per_node(self.actor, p, x)

        def apply_critic(p, x):
            return apply_shared(self.critic, p, x)

        def apply_feature(p, x):
            return apply_shared(self.feature_extractor, p, x)

        def apply_message_passer(p, x):
            assert self.message_passer is not None
            return jax.vmap(lambda x_in: self.message_passer.apply(p, x_in))(x)

        self._ppo = PPO(
            self.ppo,
            apply_sensor,
            apply_actor,
            apply_critic,
            apply_feature,
            apply_message_passer if self.message_passer is not None else None,
        )

        self.agent_state = self._init_agent_state()

        self.episode_stats = self._init_episode_stats()

        self._init_random()

    def _init_random(self):
        self.logger.info(f"[RANDOM]: Setting random seed to {self.experiment.seed}")

        random.seed(self.experiment.seed)
        np.random.seed(self.experiment.seed)

    def _init_agent(self):
        self.logger.info("[AGENT]: Initializing agent...")
        agent_indices = [0, 1, 2, 3, 4]
        match self.morph_mode:
            case MorphMode.CENTRALIZED:
                needed_copies = 1
            case MorphMode.FULLY_CONNECTED | MorphMode.RING:
                agent_mask = self.segments_per_arm > 0
                agent_indices = jnp.where(agent_mask)[0]
                needed_copies = jnp.where(self.segments_per_arm > 0, 1, 0).sum().item()
            case MorphMode.SEGMENT:
                agent_mask = self.segments_per_arm > 0
                agent_indices = jnp.where(agent_mask)[0]
                needed_copies = (
                    self.segments_per_arm.sum() + jnp.where(self.segments_per_arm > 0, 1, 0).sum()
                ).item()

        # scale actor output with size of model --> more models ==> less actions needed per model
        actor = Actor(action_dim=self.env.single_action_space.shape[0] // needed_copies)
        sensor = GenericDenseLayersWithActivation(layer_sizes=[300, 300, 300])
        message_passer: Optional[nn.Module] = (
            MessagePasser(
                hidden_dim=300,
                num_propagation_steps=self.cfg.architecture.message_passing_steps or 4,
                adj_matrix=self.adj,
            )
            if self.morph_mode != MorphMode.CENTRALIZED
            else None
        )

        feature_extractor = GenericDenseLayersWithActivation(layer_sizes=[300, 300, 300])
        critic = OneDenseLayerMLP()
        return (
            sensor,
            message_passer,
            actor,
            feature_extractor,
            critic,
            needed_copies,
            agent_indices,
        )

    def _init_agent_state(self) -> TrainState:
        self.logger.info("[AGENT STATE]: Initializing agent state...")

        self.key, sensor_key, actor_key, critic_key, feature_extractor_key, message_passer_key = (
            jax.random.split(self.key, 6)
        )

        dummy_reset = self.env.reset(seed=0)

        for k, v in dummy_reset.observations.items():
            self.logger.debug(k, v.shape)

        sample_obs = self.obs_processor(dummy_reset.observations)[0]  # take first env

        self.logger.debug(f"[_init_agent_state] sample_obs: {sample_obs.shape}")
        self.obs_mean = jnp.zeros((sample_obs.shape[-1],))
        self.obs_var = jnp.ones((sample_obs.shape[-1],))
        self.obs_count = 1e-4
        self.logger.debug(f"[_init_agent_state] obs_mean: {self.obs_mean.shape}")
        self.logger.debug(f"[_init_agent_state] obs_var: {self.obs_var.shape}")

        self.logger.debug(f"[_init_agent_state]: Needed copies: {self.needed_copies}")
        sensor_keys = jax.random.split(sensor_key, self.needed_copies)
        actor_keys = jax.random.split(actor_key, self.needed_copies)

        # (needed_copies, X)
        sensor_params = jax.vmap(lambda k: self.sensor.init(k, sample_obs))(sensor_keys)
        self.logger.debug(
            f"[_init_agent_state] sensor_params: {jax.tree.map(lambda x: x.shape, sensor_params)}"
        )

        single_sensor_param = jax.tree.map(lambda x: x[0], sensor_params)
        self.logger.debug(
            f"[_init_agent_state] single_sensor_param: {
                jax.tree.map(lambda x: x.shape, single_sensor_param)
            }"
        )

        sensor_params_sample = self.sensor.apply(single_sensor_param, sample_obs)
        self.logger.debug(
            f"[_init_agent_state] sensor_params_sample shape: {sensor_params_sample.shape}"
        )

        actor_params = jax.vmap(lambda k: self.actor.init(k, sensor_params_sample))(actor_keys)
        self.logger.debug(
            f"[_init_agent_state] actor_params: {jax.tree.map(lambda x: x.shape, actor_params)}"
        )

        message_passer_params = {}
        if self.morph_mode != MorphMode.CENTRALIZED:
            assert self.message_passer is not None, "decentralized modes require a message passer"

            message_passer_params = self.message_passer.init(
                message_passer_key,
                self.sensor.apply(single_sensor_param, sample_obs),
            )
            self.logger.debug(
                f"[_init_agent_state] message_passer_params: {
                    jax.tree.map(lambda x: x.shape, message_passer_params)
                }"
            )

        flat_obs = sample_obs.reshape(-1)  # BECAUSE 1 centralized critic
        self.logger.debug(f"[_init_agent_state] flat_obs: {flat_obs.shape}")

        feature_extractor_params = self.feature_extractor.init(feature_extractor_key, flat_obs)
        self.logger.debug(
            f"[_init_agent_state] feature_extractor_params: {
                jax.tree.map(lambda x: x.shape, feature_extractor_params)
            }"
        )

        critic_input = self.feature_extractor.apply(feature_extractor_params, flat_obs)
        self.logger.debug(f"[_init_agent_state] critic_input: {critic_input.shape}")

        critic_params = self.critic.init(critic_key, critic_input)
        self.logger.debug(
            f"[_init_agent_state] critic_params: {jax.tree.map(lambda x: x.shape, critic_params)}"
        )

        return TrainState.create(
            apply_fn=None,
            params=asdict(
                AgentParams(
                    sensor_params,
                    actor_params,
                    critic_params,
                    feature_extractor_params,
                    message_passer_params,
                )
            ),
            tx=optax.chain(
                optax.clip_by_global_norm(self.ppo.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=partial(
                        _linear_schedule,
                        minibatch_count=self.ppo.num_minibatches,
                        update_epochs=self.ppo.update_epochs,
                        num_iterations=self.num_iterations,
                        learning_rate=self.ppo.learning_rate,
                    )
                    if self.ppo.anneal_lr
                    else self.ppo.learning_rate,
                    eps=1e-5,
                ),
            ),
        )

    def _init_episode_stats(self) -> EpisodeStatistics:
        self.logger.info("[EPISODE STATS]: Initializing episode stats...")

        return EpisodeStatistics(
            episode_returns=jnp.zeros(self.ppo.num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(self.ppo.num_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(self.ppo.num_envs, jnp.float32),
            returned_episode_lengths=jnp.zeros(self.ppo.num_envs, dtype=jnp.int32),
        )

    def _rollout(self, env_state, next_obs, next_done) -> tuple[Any, ...]:
        return self._rollout_jit(
            self.agent_state,
            self.episode_stats,
            env_state,
            next_obs,
            next_done,
            self.key,
        )

    def _compute_gae(self, storage, next_obs, next_done) -> Storage:
        return self._compute_gae_jit(
            self.agent_state,
            storage,
            next_obs,
            next_done,
        )

    def _log(
        self,
        global_step,
        episode_stats,
        start_time,
        iteration_time_start,
        training_measurements,
        storage,
    ):
        data = jax.device_get(
            {
                "rewards": storage.rewards,
                "values": storage.values,
                "returns": storage.returns,
                "advantages": storage.advantages,
            }
        )

        rollout_metrics = {
            "rollout/reward_mean": float(np.mean(data["rewards"])),
            "rollout/return_mean": float(np.mean(data["returns"])),
            "rollout/value_mean": float(np.mean(data["values"])),
            "rollout/advantage_mean": float(np.mean(data["advantages"])),
            "rollout/advantage_std": float(np.std(data["advantages"])),
            "rollout/value_vs_return_mse": float(np.mean((data["values"] - data["returns"]) ** 2)),
        }

        metrics = {
            "charts/episodic_return": training_measurements.avg_episodic_return,
            "charts/episodic_length": float(
                np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            ),
            "charts/explained_variance": training_measurements.explained_variance,
            "losses/value_loss": training_measurements.v_loss[-1, -1].item(),
            "losses/policy_loss": training_measurements.pg_loss[-1, -1].item(),
            "losses/entropy": training_measurements.entropy_loss[-1, -1].item(),
            "losses/approx_kl": training_measurements.approx_kl[-1, -1].item(),
            "charts/learning_rate": self.agent_state.opt_state[1]
            .hyperparams["learning_rate"]
            .item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/SPS_update": int(
                self.ppo.num_envs * self.ppo.num_steps / (time.time() - iteration_time_start)
            ),
            "termi_trunci/num_terminated": training_measurements.num_terminated,
            "termi_trunci/num_truncated": training_measurements.num_truncated,
            "termi_trunci/avg_terminated_ep_length": training_measurements.avg_terminated_length,
            "termi_trunci/avg_truncated_ep_length": training_measurements.avg_truncated_length,
            **rollout_metrics,
        }

        self.logger.log(metrics, step=global_step)

    def _step(self, env_state, next_obs, next_done, iteration: int) -> tuple:
        if iteration == 1:
            self.logger.log_non_interactive(f"Starting first rollout (JIT): {time.ctime()}")
        self.logger.debug(f"[_step] next_obs (in): {next_obs.shape}")
        (
            self.agent_state,
            self.episode_stats,
            next_obs,
            next_done,
            storage,
            self.key,
            next_env_state,
            terminated_any,
            truncated_any,
        ) = self._rollout(env_state, next_obs, next_done)
        self.logger.debug(f"[_step] next_obs (post-rollout): {next_obs.shape}")
        if iteration == 1:
            self.logger.log_non_interactive(f"First rollout completed: {time.ctime()}")

        storage = self._compute_gae(storage, next_obs, next_done)
        self.logger.debug(f"[_step] storage.obs (post-gae): {storage.obs.shape}")
        if iteration == 1:
            self.logger.log_non_interactive(f"Starting first PPO update (JIT): {time.ctime()}")

        self.agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, self.key = (
            self._ppo.update_ppo(self.agent_state, storage, self.key)
        )

        if iteration == 1:
            self.logger.log_non_interactive(f"First PPO update completed: {time.ctime()}")

        avg_episodic_return = float(
            jnp.mean(jax.device_get(self.episode_stats.returned_episode_returns)).item()
        )

        explained_var = _compute_explained_variance(storage.values, storage.returns)

        terminated = terminated_any
        truncated = truncated_any
        episode_lengths = self.episode_stats.returned_episode_lengths

        num_terminated = int(jnp.sum(terminated).item())
        num_truncated = int(jnp.sum(truncated).item())

        avg_terminated_length = jnp.sum(episode_lengths * terminated) / jnp.maximum(
            jnp.sum(terminated), 1
        )

        avg_truncated_length = jnp.sum(episode_lengths * truncated) / jnp.maximum(
            jnp.sum(truncated), 1
        )

        return (
            next_env_state,
            next_obs,
            next_done,
            TrainingMeasurements(
                loss=loss,
                pg_loss=pg_loss,
                v_loss=v_loss,
                entropy_loss=entropy_loss,
                approx_kl=approx_kl,
                avg_episodic_return=avg_episodic_return,
                explained_variance=explained_var,
                num_terminated=num_terminated,
                num_truncated=num_truncated,
                avg_terminated_length=avg_terminated_length,
                avg_truncated_length=avg_truncated_length,
            ),
            storage,
        )

    def _close(self):
        self.env.close()

    def _save_model(self, model_path: str):
        self.logger.info("[SAVE]: Saving the final model...")
        self.logger.save_final_model(params=self.agent_state.params, metadata=asdict(self.cfg))

    def _save_checkpoint(self, iteration: int):
        self.logger.info(f"[SAVE]: Saving checkpoint at iteration {iteration}...")
        self.logger.save_checkpoint(
            params=self.agent_state.params, step=iteration, metadata=asdict(self.cfg)
        )

    def train(self):
        """
        Train the PPO agent for a specified number of iterations.
        Closes the environment at the end of training.
        """
        self.logger.info(f"running name: {self.run_name}")

        self.logger.info("[TRAIN]: Resetting environment...")
        self.logger.log_non_interactive(f"Initial reset started: {time.ctime()}")

        env_state = self.env.reset(seed=self.experiment.seed)

        next_obs = self.obs_processor(env_state.observations)
        self.logger.debug(f"[train] next_obs: {next_obs.shape}")

        next_done = jnp.zeros(self.ppo.num_envs, dtype=jnp.bool_)

        self.logger.log_non_interactive(f"Initial reset completed: {time.ctime()}")

        global_step = 0
        start_time = time.time()

        iter_bar = self.logger.progress_bar(range(1, self.num_iterations + 1))
        for iteration in iter_bar:
            iteration_time_start = time.time()

            env_state, next_obs, next_done, training_measurements, storage = self._step(
                env_state, next_obs, next_done, iteration=iteration
            )

            global_step += self.ppo.num_steps * self.ppo.num_envs
            self._log(
                global_step,
                self.episode_stats,
                start_time,
                iteration_time_start,
                training_measurements,
                storage,
            )

            sps = int(global_step / (time.time() - start_time))
            remaining_steps = self.ppo.total_timesteps - global_step
            eta_seconds = int(remaining_steps / sps) if sps > 0 else 0
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

            self.logger.log_non_interactive(
                f"Iteration {iteration}/{self.num_iterations} | "
                f"Step {global_step}/{self.ppo.total_timesteps} | "
                f"SPS {sps} | "
                f"Return {training_measurements.avg_episodic_return:.4f} | "
                f"ETA {eta_str}"
            )

            if self.logging_cfg.save_checkpoints and self.logging_cfg.checkpoint_frequency > 0:
                if iteration % self.logging_cfg.checkpoint_frequency == 0:
                    self._save_checkpoint(iteration)

            if getattr(self.cfg.experiment, "debug_sanity", False):
                self.logger.info("\n[SANITY CHECK] Successfully completed 1 epoch")
                break

        if self.logging_cfg.save_model:
            model_path = f"{self.run_dir}/{self.experiment.exp_name}.cleanrl_model"
            self._save_model(model_path=model_path)

        self._close()
