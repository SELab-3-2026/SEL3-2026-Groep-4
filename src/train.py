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
from brittle_star_project.rl import Network, Actor, Critic, AgentParams, Storage


def convert_obs_dict_to_array(obs_dict: dict) -> jnp.ndarray:
    return jax.vmap(lambda o: jnp.concatenate([v.flatten() for v in o.values() if v.size > 0]))(
        obs_dict
    )


def make_env(num_envs: int) -> Callable:
    def thunk():
        return BrittleStarJaxEnvWrapper.default(num_envs=num_envs)

    return thunk


def train(args: PPOArgs):
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = 10 if True else args.total_timesteps // args.batch_size
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
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Running on device: {device}")

    print("Creating the environment...")
    env = make_env(num_envs=args.num_envs)()

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
    network = Network()
    actor = Actor(action_dim=env.single_action_space.shape[0])  # continuous actions for MJX
    critic = Critic()

    sample_obs = jnp.concatenate(
        [
            v.flatten()
            for v in env.single_observation_space.sample(rng=jax.random.PRNGKey(0)).values()
            if v.size > 0
        ]
    )
    network_params = network.init(network_key, sample_obs)
    actor_params = actor.init(actor_key, network.apply(network_params, sample_obs))
    critic_params = critic.init(critic_key, network.apply(network_params, sample_obs))

    agent_state = TrainState.create(
        apply_fn=None,
        params=asdict(AgentParams(network_params, actor_params, critic_params)),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )

    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(agent_state.params["network_params"], next_obs)
        # Continuous actions: sample from a Gaussian parameterized by the actor
        mean, log_std = actor.apply(agent_state.params["actor_params"], hidden)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=mean.shape)
        std = jnp.exp(log_std)
        action = mean + noise * std
        logprob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(-1)
        value = critic.apply(agent_state.params["critic_params"], hidden)
        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: jnp.ndarray,
        action: np.ndarray,
    ):
        hidden = network.apply(params["network_params"], x)
        mean, log_std = actor.apply(params["actor_params"], hidden)
        std = jnp.exp(log_std)
        logprob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(-1)
        entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).sum(-1)
        value = critic.apply(params["critic_params"], hidden).squeeze(-1)
        return logprob, entropy, value

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
            network.apply(agent_state.params["network_params"], next_obs),
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

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(agent_state, storage, key):
        def update_epoch(carry, _):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x):
                x = jax.random.permutation(subkey, x)
                return jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, metrics = jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
            return (agent_state, key), metrics

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

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
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

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
    for _ in iters_bar:
        iteration_time_start = time.time()

        agent_state, episode_stats, next_obs, next_done, storage, key, next_env_state = rollout(
            agent_state, episode_stats, next_obs, next_done, key, next_env_state
        )

        global_step += args.num_steps * args.num_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state, storage, key
        )

        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        iters_bar.set_postfix_str(
            f"global_step={global_step}, avg_episodic_return={avg_episodic_return}"
        )

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
        print(f"model saved to {model_path}")

    env.close()
    writer.close()


def main() -> None:
    args = tyro.cli(PPOArgs)
    train(args)


if __name__ == "__main__":
    main()
