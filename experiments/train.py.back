import random
import time
from dataclasses import asdict
from functools import partial
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm
import tyro
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from brittle_star_project.dataclasses import PPOArgs
from brittle_star_project.dataclasses.EpisodeStatistics import EpisodeStatistics
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from MLPs.mlps import (
    GenericDenseLayersWithActivation,
    Actor,
    OneDenseLayerMLP,
    AgentParams,
    Storage,
)
from plots import simple_plot
from ppo import PPO


def convert_obs_dict_to_array(obs_dict: dict) -> jnp.ndarray:
    return jax.vmap(lambda o: jnp.concatenate([v.flatten() for v in o.values() if v.size > 0]))(
        obs_dict
    )


def make_env(config_path: str | None, num_envs: int) -> Callable:
    def thunk():
        if config_path is None:
            return BrittleStarJaxEnvWrapper.default(num_envs=num_envs)
        return BrittleStarJaxEnvWrapper.from_config(config_path, num_envs=num_envs)

    return thunk

def save_model(model_path: str, agent_state: TrainState, args: PPOArgs):
    with open(model_path, "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                [
                    vars(args),
                    [
                        agent_state.params["network_params"],
                        agent_state.params["actor_params"],
                        agent_state.params["critic_params"],
                    ],
                ]
            )
        )


def train(args: PPOArgs):
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    # args.num_iterations = args.total_timesteps // args.batch_size
    args.num_iterations = 5
    run_name = f"{args.exp_name}__seed_{args.seed}__{int(time.time())}"
    print(f"running name: {run_name}")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|---|---|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, sensor_key, actor_key, critic_key, feature_extractor_key = jax.random.split(key, 5)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Running on device: {device}")

    print("Creating the environment...")
    env = make_env(config_path=args.config_path, num_envs=args.num_envs)()
    print(f"Environment: {env}")

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    def step_env_wrapped(episode_stats: EpisodeStatistics, env_state, action):
        next_env_state = env.step(env_state, action)

        # Extract per-environment signals from the state object
        reward = next_env_state.reward  # (num_envs,)
        terminated = next_env_state.terminated  # (num_envs,)
        truncated = next_env_state.truncated  # (num_envs,)
        done = terminated | truncated  # (num_envs,)

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
            (convert_obs_dict_to_array(next_env_state.observations), reward, done),
        )

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    print("Initializing the models...")
    sensor = GenericDenseLayersWithActivation()
    feature_extractor = GenericDenseLayersWithActivation()
    actor = Actor(action_dim=env.single_action_space.shape[0])  # continuous actions for MJX
    critic = OneDenseLayerMLP()
    # messager = OneDenseLayerMLP()

    sample_obs = jnp.concatenate(
        [
            v.flatten()
            for v in env.single_observation_space.sample(rng=jax.random.PRNGKey(0)).values()
            if v.size > 0
        ]
    )
    sensor_params = sensor.init(sensor_key, sample_obs)
    feature_extractor_params = feature_extractor.init(feature_extractor_key, sample_obs)
    actor_params = actor.init(actor_key, sensor.apply(sensor_params, sample_obs))
    critic_params = critic.init(
        critic_key, feature_extractor.apply(feature_extractor_params, sample_obs)
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=asdict(
            AgentParams(sensor_params, actor_params, critic_params, feature_extractor_params)
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )

    sensor.apply = jax.jit(sensor.apply)
    feature_extractor.apply = jax.jit(feature_extractor.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    ppo_instance = PPO(args, sensor, actor, critic, feature_extractor)

    @jax.jit
    def get_action_and_value_noise(
        agent_state: TrainState,
        next_obs: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = sensor.apply(agent_state.params["sensor_params"], next_obs)
        hidden_critic = feature_extractor.apply(
            agent_state.params["feature_extractor_params"], next_obs
        )

        # Continuous actions: sample from a Gaussian parameterized by the actor
        mean, log_std = actor.apply(agent_state.params["actor_params"], hidden)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=mean.shape)
        std = jnp.exp(log_std)
        action = mean + noise * std
        logprob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(-1)
        value = critic.apply(agent_state.params["critic_params"], hidden_critic)
        return action, logprob, value.squeeze(-1), key


    # GAE
    @jax.jit
    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone
        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    @jax.jit
    def compute_gae(agent_state, next_obs, next_done, storage):
        next_value = critic.apply(
            agent_state.params["critic_params"],
            sensor.apply(agent_state.params["sensor_params"], next_obs),
        ).squeeze(-1)

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda),
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        return storage.replace(advantages=advantages, returns=advantages + storage.values)
    # END GAE

    # --- Main training loop ---
    global_step = 0
    start_time = time.time()

    # Reset once to get initial state
    print("Resetting the environment...")
    next_env_state = env.reset(seed=args.seed)
    next_obs = convert_obs_dict_to_array(next_env_state.observations)
    next_done = jnp.zeros(args.num_envs, dtype=jnp.bool_)

    def step_once(carry, _, env_step_fn):
        agent_state, episode_stats, obs, done, key, env_state = carry
        action, logprob, value, key = get_action_and_value_noise(agent_state, obs, key)

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

    def rollout(
        agent_state, episode_stats, next_obs, next_done, key, env_state, step_once_fn, max_steps
    ):
        (agent_state, episode_stats, next_obs, next_done, key, env_state), storage = jax.lax.scan(
            step_once_fn,
            (agent_state, episode_stats, next_obs, next_done, key, env_state),
            (),
            max_steps,
        )
        return agent_state, episode_stats, next_obs, next_done, storage, key, env_state

    rollout = partial(
        rollout,
        step_once_fn=partial(step_once, env_step_fn=step_env_wrapped),
        max_steps=args.num_steps,
    )

    print("Starting training...")
    iters_bar = tqdm.tqdm(range(1, args.num_iterations + 1))
    returns = []
    for _ in iters_bar:
        iteration_time_start = time.time()

        agent_state, episode_stats, next_obs, next_done, storage, key, next_env_state = rollout(
            agent_state, episode_stats, next_obs, next_done, key, next_env_state
        )

        global_step += args.num_steps * args.num_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = ppo_instance.update_ppo(
            agent_state, storage, key
        )

        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        iters_bar.set_postfix_str(
            f"global_step={global_step}, avg_episodic_return={avg_episodic_return}"
        )

        returns.append(avg_episodic_return)

        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length",
            np.mean(jax.device_get(episode_stats.returned_episode_lengths)),
            global_step,
        )
        writer.add_scalar(
            "charts/learning_rate",
            agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)

        # iters_bar.set_postfix_str(f"SPS: {int(global_step / (time.time() - start_time))}")

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update",
            int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)),
            global_step,
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        save_model(model_path, agent_state, args)
        print(f"model saved to {model_path}")

    env.close()
    writer.close()

    print("Saving loss plot...")
    simple_plot(
        list(range(len(returns))),
        returns,
        show_window=True,
        filename=f"runs/{run_name}/{args.exp_name}_losses.png",
    )


def main() -> None:
    args = tyro.cli(PPOArgs)
    train(args)


if __name__ == "__main__":
    main()
