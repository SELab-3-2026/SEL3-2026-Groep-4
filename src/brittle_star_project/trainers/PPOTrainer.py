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

from brittle_star_project.dataclasses import EpisodeStatistics, PPOArgs
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from brittle_star_project.MLPs.mlps import (
    Actor,
    AgentParams,
    GenericDenseLayersWithActivation,
    OneDenseLayerMLP,
    Storage,
)
from brittle_star_project.ppo import PPO

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
def _convert_obs_dict_to_array(obs_dict: dict) -> jnp.ndarray:
    return jax.vmap(lambda o: jnp.concatenate([v.flatten() for v in o.values() if v.size > 0]))(
        obs_dict
    )


def _get_action_and_value_noise(
    sensor: GenericDenseLayersWithActivation,
    feature_extractor: GenericDenseLayersWithActivation,
    actor: Actor,
    critic: OneDenseLayerMLP,
    agent_state: TrainState,
    next_obs: jnp.ndarray,
    key: jax.random.PRNGKey,
    action_low,
    action_high
):
    hidden = sensor.apply(agent_state.params["sensor_params"], next_obs)
    hidden_critic = feature_extractor.apply(
        agent_state.params["feature_extractor_params"], next_obs
    )

    mean, log_std = actor.apply(agent_state.params["actor_params"], hidden)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=mean.shape)
    std = jnp.exp(log_std)
    action = mean + noise * std
    clipped_action = _clip_action(
        action,
        action_low,
        action_high
    )
    logprob = -0.5 * (((clipped_action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(-1)
    value = critic.apply(agent_state.params["critic_params"], hidden_critic)
    return clipped_action, logprob, value.squeeze(-1), key


def _step_once(
    carry,
    _,
    env_step_fn,
    sensor: GenericDenseLayersWithActivation,
    feature_extractor: GenericDenseLayersWithActivation,
    actor: Actor,
    critic: OneDenseLayerMLP,
    action_low,
    action_high
):
    agent_state, episode_stats, obs, done, key, env_state = carry
    action, logprob, value, key = _get_action_and_value_noise(
        sensor, feature_extractor, actor, critic, agent_state, obs, key, action_low, action_high
    )

    episode_stats, env_state, (next_obs, reward, next_done) = env_step_fn(
        episode_stats, env_state, action
    )

    storage = Storage(
        obs=obs,
        actions=action,
        logprobs=logprob,
        dones=done,
        values=value,
        rewards=reward,
        returns=jnp.zeros_like(reward),
        advantages=jnp.zeros_like(reward),
    )
    return (agent_state, episode_stats, next_obs, next_done, key, env_state), storage


def _step_env_wrapped(episode_stats, env_state, action, env_step_fn):
    next_env_state = env_step_fn(env_state, action)

    reward = next_env_state.reward
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
        (_convert_obs_dict_to_array(next_env_state.observations), reward, done),
    )


def _rollout_jit(
    agent_state,
    episode_stats,
    env_state,
    next_obs,
    next_done,
    key,
    max_steps,
    step_env_fn,
    sensor: GenericDenseLayersWithActivation,
    feature_extractor: GenericDenseLayersWithActivation,
    actor: Actor,
    critic: OneDenseLayerMLP,
    action_low,
    action_high
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
            action_high=action_high
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
    return storage.replace(advantages=advantages, returns=advantages + storage.values)


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
    def __init__(self, args: PPOArgs, env: BrittleStarJaxEnvWrapper, run_dir: str, run_name: str):
        self.args = args
        self.env = env
        self.run_dir = run_dir
        self.run_name = run_name
        self.logger = get_logger()

        self.key = jax.random.PRNGKey(args.seed)

        self.sensor, self.feature_extractor, self.actor, self.critic = self._init_agent()
        self.sensor.apply = jax.jit(self.sensor.apply)
        self.feature_extractor.apply = jax.jit(self.feature_extractor.apply)
        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        action_low = jnp.asarray(self.env.single_action_space.low, dtype=jnp.float32)
        action_high = jnp.asarray(self.env.single_action_space.high, dtype=jnp.float32)

        self._rollout_jit = jax.jit(
            partial(
                _rollout_jit,
                max_steps=self.args.num_steps,
                step_env_fn=partial(_step_env_wrapped, env_step_fn=self.env.step),
                sensor=self.sensor,
                feature_extractor=self.feature_extractor,
                actor=self.actor,
                critic=self.critic,
                action_low=action_low,
                action_high=action_high
            )
        )
        self._compute_gae_jit = jax.jit(
            partial(
                _compute_gae_jit,
                num_envs=self.args.num_envs,
                gamma=self.args.gamma,
                gae_lambda=self.args.gae_lambda,
                feature_extractor=self.feature_extractor,
                critic=self.critic,
            )
        )

        self._ppo = PPO(self.args, self.sensor, self.actor, self.critic, self.feature_extractor)

        self.agent_state = self._init_agent_state()

        self.episode_stats = self._init_episode_stats()

        self._init_random()

    def _init_random(self):
        self.logger.info(f"[RANDOM]: Setting random seed to {self.args.seed}")

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

    def _init_agent(self):
        self.logger.info("[AGENT]: Initializing agent...")

        sensor = GenericDenseLayersWithActivation()
        feature_extractor = GenericDenseLayersWithActivation()
        actor = Actor(action_dim=self.env.single_action_space.shape[0])
        critic = OneDenseLayerMLP()
        return sensor, feature_extractor, actor, critic

    def _init_agent_state(self) -> TrainState:
        self.logger.info("[AGENT STATE]: Initializing agent state...")

        self.key, sensor_key, actor_key, critic_key, feature_extractor_key = jax.random.split(
            self.key, 5
        )

        sample_obs = jnp.concatenate(
            [
                v.flatten()
                for v in self.env.single_observation_space.sample(
                    rng=jax.random.PRNGKey(0)
                ).values()
                if v.size > 0
            ]
        )
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
                optax.clip_by_global_norm(self.args.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=partial(
                        _linear_schedule,
                        minibatch_count=self.args.num_minibatches,
                        update_epochs=self.args.update_epochs,
                        num_iterations=self.args.num_iterations,
                        learning_rate=self.args.learning_rate,
                    )
                    if self.args.anneal_lr
                    else self.args.learning_rate,
                    eps=1e-5,
                ),
            ),
        )

    def _init_episode_stats(self) -> EpisodeStatistics:
        self.logger.info("[EPISODE STATS]: Initializing episode stats...")

        return EpisodeStatistics(
            episode_returns=jnp.zeros(self.args.num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(self.args.num_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(self.args.num_envs, jnp.float32),
            returned_episode_lengths=jnp.zeros(self.args.num_envs, dtype=jnp.int32),
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
    ):
        metrics = {
            "charts/avg_episodic_return": training_measurements.avg_episodic_return,
            "charts/avg_episodic_length": np.mean(
                jax.device_get(episode_stats.returned_episode_lengths)
            ),
            "charts/learning_rate": self.agent_state.opt_state[1]
            .hyperparams["learning_rate"]
            .item(),
            "charts/explained_variance": training_measurements.explained_variance,
            "charts/num_terminated": training_measurements.num_terminated,
            "charts/num_truncated": training_measurements.num_truncated,
            "charts/avg_terminated_ep_length": training_measurements.avg_terminated_length,
            "charts/avg_truncated_ep_length": training_measurements.avg_truncated_length,
            "losses/value_loss": training_measurements.v_loss[-1, -1].item(),
            "losses/policy_loss": training_measurements.pg_loss[-1, -1].item(),
            "losses/entropy": training_measurements.entropy_loss[-1, -1].item(),
            "losses/approx_kl": training_measurements.approx_kl[-1, -1].item(),
            "losses/loss": training_measurements.loss[-1, -1].item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/SPS_update": int(
                self.args.num_envs * self.args.num_steps / (time.time() - iteration_time_start)
            ),
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
        )

    def _close(self):
        self.env.close()

    def _save_model(self, model_path: str):
        self.logger.info("[SAVE]: Saving the final model...")

        params = [
            vars(self.args),
            [
                self.agent_state.params["sensor_params"],
                self.agent_state.params["actor_params"],
                self.agent_state.params["critic_params"],
                self.agent_state.params["feature_extractor_params"],
            ],
        ]
        self.logger.save_final_model(params=params)

    def train(self):
        """
        Train the PPO agent for a specified number of iterations
        (passed through PPOArgs in constructor).
        Closes the environment at the end of training.
        """
        self.logger.info(f"running name: {self.run_name}")

        self.logger.info("[TRAIN]: Resetting environment...")
        self.logger.log_non_interactive(f"Initial reset started: {time.ctime()}")

        env_state = self.env.reset(seed=self.args.seed)
        next_obs = _convert_obs_dict_to_array(env_state.observations)
        next_done = jnp.zeros(self.args.num_envs, dtype=jnp.bool_)

        self.logger.log_non_interactive(f"Initial reset completed: {time.ctime()}")

        global_step = 0
        start_time = time.time()

        iter_bar = self.logger.progress_bar(range(1, self.args.num_iterations + 1))
        for iteration in iter_bar:
            iteration_time_start = time.time()

            env_state, next_obs, next_done, training_measurements = self._step(
                env_state, next_obs, next_done, iteration=iteration
            )

            global_step += self.args.num_steps * self.args.num_envs
            self._log(
                global_step,
                self.episode_stats,
                start_time,
                iteration_time_start,
                training_measurements,
            )

            sps = int(global_step / (time.time() - start_time))
            remaining_steps = self.args.total_timesteps - global_step
            eta_seconds = int(remaining_steps / sps) if sps > 0 else 0
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

            self.logger.log_non_interactive(
                f"Iteration {iteration}/{self.args.num_iterations} | "
                f"Step {global_step}/{self.args.total_timesteps} | "
                f"SPS {sps} | "
                f"Return {training_measurements.avg_episodic_return:.4f} | "
                f"ETA {eta_str}"
            )

        if self.args.save_model:
            model_path = f"{self.run_dir}/{self.args.exp_name}.cleanrl_model"
            self._save_model(model_path=model_path)

        self._close()
