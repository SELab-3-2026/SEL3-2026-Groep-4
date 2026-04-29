import datetime
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from experiment_logger import get_logger

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.dataclasses import EpisodeStatistics
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from brittle_star_project.MLPs.mlps import (
    Actor,
    AgentParams,
    GenericDenseLayersWithActivation,
    OneDenseLayerMLP,
    Storage,
)
from brittle_star_project.ppo import PPO
from brittle_star_project.environment import MorphMode

# TODO: move to config
_ALLOWED_OBS_KEYS = {
    "joint_position",
    "joint_velocity",
    "joint_actuator_force",
    "actuator_force",
    "disk_position",
    "disk_rotation",
    "disk_linear_velocity",
    "disk_angular_velocity",
    "unit_xy_direction_to_target",
    "xy_distance_to_target",
}
# TODO: clip scaled reward?


def build_adjacency(segments_per_arm, mode: MorphMode):
    num_arms = sum(1 for s in segments_per_arm if s > 0)
    num_segments = sum(segments_per_arm)

    # FOR NOW SEMI HARDCODE:
    # CENTRALIZED: 1 agent, no stress, adja = 1,1 = [[1]]
    # FULLY CONNECTED: 5 agents: adj = alle 1
    # CENTRAL DISK:#arms= 5 agents, only neighbor as adjacent so diagonal kinda..
    # ARM = #segments agents: diago kinda, but extra, center ring too, put center mlps first or..

    if mode == MorphMode.CENTRALIZED:
        return jnp.ones((1, 1))

    if mode == MorphMode.FULLY_CONNECTED:
        adj = jnp.ones((num_arms, num_arms))  # everybody adjacent everybody
        return adj

    if mode == MorphMode.RING:  # ring
        adj = jnp.zeros((num_arms, num_arms))
        for i in range(num_arms):
            adj = adj.at[i, i].set(1)  # self
            adj = adj.at[i, (i - 1) % num_arms].set(1)
            adj = adj.at[i, (i + 1) % num_arms].set(1)  # left and right..
        return adj

    if mode == MorphMode.SEGMENT:
        num_nodes = num_arms + num_segments
        adj = jnp.zeros((num_nodes, num_nodes))

        # first ring
        for i in range(num_arms):
            # self
            adj = adj.at[i, i].set(1)

            # ring neighbors
            adj = adj.at[i, (i - 1) % num_arms].set(1)
            adj = adj.at[i, (i + 1) % num_arms].set(1)

        # then segment chains
        idx = 0
        for arm_idx, seg_count in enumerate(segments_per_arm):
            for i in range(seg_count):
                seg_node = num_arms + idx + i

                adj = adj.at[seg_node, seg_node].set(1)
                if i > 0:
                    adj = adj.at[seg_node, seg_node - 1].set(1)
                if i < seg_count - 1:
                    adj = adj.at[seg_node, seg_node + 1].set(1)

            idx += seg_count

        idx = 0
        for arm_idx, seg_count in enumerate(segments_per_arm):
            first_seg = num_arms + idx  # first segment of this arm

            # connect ring node first segment
            adj = adj.at[arm_idx, first_seg].set(1)
            adj = adj.at[first_seg, arm_idx].set(1)

            idx += seg_count

        return adj


@jax.jit
def _clip_action(action: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(action, low, high)


def _compute_explained_variance(values: jnp.ndarray, returns: jnp.ndarray) -> float:
    var_returns = jnp.var(returns)
    explained_var = 1.0 - jnp.var(returns - values) / (var_returns + 1e-8)
    return float(explained_var)


@jax.jit
def _linear_schedule(count, minibatch_count, update_epochs, num_iterations, learning_rate):
    frac = 1.0 - (count // (minibatch_count * update_epochs)) / num_iterations
    return learning_rate * frac


@jax.jit
def _normalize_obs(obs, mean, var, eps=1e-8):
    return jnp.clip((obs - mean) / jnp.sqrt(var + eps), -10.0, 10.0)


@jax.jit
def _convert_obs_dict_to_array_morphology(obs_dict, morph_mode, segments_per_arm):
    num_segments = sum(segments_per_arm)
    num_arms = sum(1 for s in segments_per_arm if s > 0)

    def _filter_and_flatten(o):
        values = []

        for key in sorted(o.keys()):
            if key not in _ALLOWED_OBS_KEYS:
                continue

            v = o[key]
            if v.size == 0:
                continue

            # -------- CENTRALIZED --------
            if morph_mode == 0:
                values.append(v.reshape(v.shape[0], -1))
                continue

            # -------- SPLIT TO SEGMENTS --------
            if key in _JOINT_SCALED_KEYS:
                if morph_mode == 3:
                    # special case: segment lvl and scale with joints,
                    # needs to split logic for center mlps and arms
                    # center:
                    # 3 joints per arm, each joint has 2 values
                    B = v.shape[0]
                    center_size = num_arms * 3 * 2
                    v_center = v[:, :center_size]
                    v_center = v_center.reshape(B, num_arms, 3, 2)

                    v_segs = v[:, center_size:]
                    v_segs = v_segs.reshape(B, -1, 2)

                    v = jnp.concatenate([v_center.reshape(B, -1, 2), v_segs], axis=1)
                    values.append(v)
                    continue
                v = v.reshape(v.shape[0], num_segments, 2)

            elif key in _SEGMENT_SCALED_KEYS:
                v = v[..., None]  # (env, segments, 1)

            else:
                # global key, share with all
                if morph_mode == 3:  # segment
                    v = jnp.repeat(v[:, None, :], num_segments + num_arms, axis=1)

                else:  # ring or fully connect
                    v = jnp.repeat(v[:, None, :], num_arms, axis=1)
                values.append(v)
                continue

            # -------- SEGMENT MODE --------
            if morph_mode == 3:
                values.append(v)
                continue

            # -------- ARM MODE --------
            v = v.reshape(v.shape[0], num_arms, -1)
            values.append(v)

        return jnp.concatenate(values, axis=-1)

    return jax.vmap(_filter_and_flatten)(obs_dict)


# Observation keys whose size scales with the number of joints (2 per segment).
_JOINT_SCALED_KEYS = frozenset(
    {  # TODO CODE SMELL
        "joint_position",
        "joint_velocity",
        "joint_actuator_force",
        "actuator_force",
    }
)

# Observation keys whose size scales with the number of segments (1 per segment).
_SEGMENT_SCALED_KEYS = frozenset(
    {
        "segment_contact",
    }
)


# TODO: update to work with extra dimension + message passing
def _get_action_and_value_noise(
    sensor: GenericDenseLayersWithActivation,
    feature_extractor: GenericDenseLayersWithActivation,
    actor: Actor,
    critic: OneDenseLayerMLP,
    agent_state: TrainState,
    next_obs: jnp.ndarray,
    key: jax.random.PRNGKey,
    action_low,
    action_high,
    adj_matrix: jnp.ndarray,
):
    hidden = apply_per_node(sensor, agent_state.params["sensor_params"], next_obs)
    hidden_critic = apply_per_node(
        feature_extractor, agent_state.params["feature_extractor_params"], next_obs
    )

    mean, log_std = apply_per_node(actor, agent_state.params["actor_params"], hidden)
    log_std = jnp.clip(log_std, -5, 2)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=mean.shape)
    std = jnp.exp(log_std)
    raw_action = mean + noise * std
    clipped_action = _clip_action(raw_action, action_low, action_high)
    logprob = -0.5 * (((raw_action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(-1)
    value = critic.apply(agent_state.params["critic_params"], hidden_critic)

    return clipped_action, raw_action, logprob, value.squeeze(-1), mean, std, key


# TODO: update to work with extra dimension + message passing
def _step_once(
    carry,
    _,
    env_step_fn,
    adj_matrix,
    sensor: GenericDenseLayersWithActivation,
    feature_extractor: GenericDenseLayersWithActivation,
    actor: Actor,
    critic: OneDenseLayerMLP,
    action_low,
    action_high,
):
    agent_state, episode_stats, obs, done, key, env_state = carry
    clipped_action, raw_action, logprob, value, mean, std, key = _get_action_and_value_noise(
        sensor,
        feature_extractor,
        actor,
        critic,
        agent_state,
        obs,
        key,
        action_low,
        action_high,
        adj_matrix,
    )

    episode_stats, env_state, (next_obs, reward, next_done) = env_step_fn(
        episode_stats, env_state, clipped_action
    )

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
    return (agent_state, episode_stats, next_obs, next_done, key, env_state), storage


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


# TODO: update to work with extra dimension + message passing
def _step_env_wrapped(episode_stats, env_state, action, env_step_fn, morph_mode, segments_per_arm):
    next_env_state = env_step_fn(env_state, action)

    reward = _reward_fn(env_state, next_env_state)
    terminated = next_env_state.terminated
    truncated = next_env_state.truncated
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
    return (
        episode_stats,
        next_env_state,
        (
            _convert_obs_dict_to_array_morphology(
                next_env_state.observations, morph_mode, segments_per_arm
            ),
            reward,
            done,
        ),
    )


def apply_per_node(net, params, x):
    # x: (batch, nodes, feat)
    return jax.vmap(lambda node_x: net.apply(params, node_x), in_axes=1, out_axes=1)(x)


# TODO: update to work with extra dimension + message passing
def _rollout_jit(
    agent_state,
    episode_stats,
    env_state,
    next_obs,
    next_done,
    adj_matrix,
    key,
    max_steps,
    step_env_fn,
    sensor: GenericDenseLayersWithActivation,
    feature_extractor: GenericDenseLayersWithActivation,
    actor: Actor,
    critic: OneDenseLayerMLP,
    action_low,
    action_high,
):
    (agent_state, episode_stats, next_obs, next_done, key, env_state), storage = jax.lax.scan(
        partial(
            _step_once,
            sensor=sensor,
            feature_extractor=feature_extractor,
            actor=actor,
            critic=critic,
            env_step_fn=step_env_fn,
            action_low=action_low,
            action_high=action_high,
            adj_matrix=adj_matrix,
        ),
        (agent_state, episode_stats, next_obs, next_done, key, env_state),
        (),
        max_steps,
    )
    return agent_state, episode_stats, next_obs, next_done, storage, key, env_state


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
    adj_matrix: jnp.ndarray,
):
    next_value = critic.apply(
        agent_state.params["critic_params"],
        feature_extractor.apply(agent_state.params["feature_extractor_params"], next_obs),
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

        self.logger.info(f"[INIT]: Used morphology mode {self.morph_mode}")
        self.adj = build_adjacency(cfg.morphology.segments_per_arm, self.morph_mode)

        self.sensor, self.feature_extractor, self.actor, self.critic = self._init_agent()
        self.sensor.apply = jax.jit(self.sensor.apply)
        self.feature_extractor.apply = jax.jit(self.feature_extractor.apply)
        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.segments_per_arm = jnp.asarray(self.cfg.morphology.segments_per_arm, dtype=jnp.int32)

        action_low = jnp.asarray(self.env.single_action_space.low, dtype=jnp.float32)
        action_high = jnp.asarray(self.env.single_action_space.high, dtype=jnp.float32)

        self._rollout_jit = jax.jit(
            partial(
                _rollout_jit,
                max_steps=self.ppo.num_steps,
                step_env_fn=partial(
                    _step_env_wrapped,
                    env_step_fn=self.env.step,
                    morph_mode=self.morph_mode,
                    segments_per_arm=self.segments_per_arm,
                ),
                sensor=self.sensor,
                feature_extractor=self.feature_extractor,
                actor=self.actor,
                critic=self.critic,
                action_low=action_low,
                action_high=action_high,
                adj_matrix=self.adj,
            )
        )
        self._compute_gae_jit = jax.jit(
            partial(
                _compute_gae_jit,
                num_envs=self.ppo.num_envs,
                gamma=self.ppo.gamma,
                gae_lambda=self.ppo.gae_lambda,
                feature_extractor=self.feature_extractor,
                critic=self.critic,
                adj_matrix=self.adj,
            )
        )

        self._ppo = PPO(self.ppo, self.sensor, self.actor, self.critic, self.feature_extractor)

        self.agent_state = self._init_agent_state()

        self.episode_stats = self._init_episode_stats()

        self._init_random()

    def _init_random(self):
        self.logger.info(f"[RANDOM]: Setting random seed to {self.experiment.seed}")

        random.seed(self.experiment.seed)
        np.random.seed(self.experiment.seed)

    def _init_agent(self):
        self.logger.info("[AGENT]: Initializing agent...")
        sensors = []
        actors = []
        message_passers = []
        needed_copies = 1
        if (self.morph_mode == MorphMode.FULLY_CONNECTED) or (self.morph_mode == MorphMode.RING):
            needed_copies = sum(1 for s in self.segments_per_arm if s > 0)
        else:
            needed_copies = sum(self.segments_per_arm) + sum(
                1 for s in self.segments_per_arm if s > 0
            )

        for _ in range(needed_copies):
            actor = Actor(action_dim=self.env.single_action_space.shape[0])
            sensor = GenericDenseLayersWithActivation(layer_sizes=[300, 300, 300])
            sensors.append(sensor)
            actors.append(actor)
            message_passers.append(OneDenseLayerMLP())

        feature_extractor = GenericDenseLayersWithActivation(layer_sizes=[300, 300, 300])
        critic = OneDenseLayerMLP()
        return sensors, message_passers, actors, feature_extractor, critic

    def _init_agent_state(self) -> TrainState:
        self.logger.info("[AGENT STATE]: Initializing agent state...")

        self.key, sensor_key, actor_key, critic_key, feature_extractor_key = jax.random.split(
            self.key, 5
        )

        dummy_reset = self.env.reset(seed=0)
        for k, v in dummy_reset.observations.items():
            self.logger.debug(k, v.shape)
        sample_obs = _convert_obs_dict_to_array_morphology(
            dummy_reset.observations,
            self.morph_mode,
            self.segments_per_arm,
        )[0]  # take first env
        self.obs_mean = jnp.zeros((len(sample_obs),))
        self.obs_var = jnp.ones((len(sample_obs),))
        self.obs_count = 1e-4
        sensor_params = self.sensor.init(sensor_key, sample_obs)
        feature_extractor_params = self.feature_extractor.init(feature_extractor_key, sample_obs)
        actor_params = self.actor.init(actor_key, self.sensor.apply(sensor_params, sample_obs))
        critic_params = self.critic.init(
            critic_key, self.feature_extractor.apply(feature_extractor_params, sample_obs)
        )

        return TrainState.create(
            apply_fn=None,
            params=asdict(
                AgentParams(sensor_params, actor_params, critic_params, feature_extractor_params)
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

    def _update_obs_stats(self, obs: jnp.ndarray):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count

        new_mean = self.obs_mean + delta * batch_count / total_count

        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        new_var = M2 / total_count

        self.obs_mean = new_mean
        self.obs_var = new_var
        self.obs_count = total_count

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

        (
            self.agent_state,
            self.episode_stats,
            next_obs,
            next_done,
            storage,
            self.key,
            next_env_state,
        ) = self._rollout(env_state, next_obs, next_done)

        if iteration == 1:
            self.logger.log_non_interactive(f"First rollout completed: {time.ctime()}")

        storage = self._compute_gae(storage, next_obs, next_done)

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

        terminated = next_env_state.terminated
        truncated = next_env_state.truncated
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
        next_obs = _convert_obs_dict_to_array_morphology(
            env_state.observations,
            self.morph_mode,
            self.segments_per_arm,
        )
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
            self._update_obs_stats(next_obs)
            next_obs = _normalize_obs(next_obs, self.obs_mean, self.obs_var)

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
