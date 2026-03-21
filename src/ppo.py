# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

@jax.jit
def get_action_and_value2(
    params: flax.core.FrozenDict,
    x: np.ndarray,
    action: np.ndarray,
):
    """calculate value, logprob of supplied `action`, and entropy"""
    hidden = network.apply(params.network_params, x)
    # assume that actor returns mean and log std over continuos action space, why log?, better for..?, research this
    mean, log_std = actor.apply(params.actor_params, hidden)
    std = jnp.exp(log_std)
    
    # compute logprob of the given action
    var = std ** 2
    logprob = -0.5 * (((action - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    logprob = logprob.sum(-1)
    
    # Validate that this is a good entropy (check other ppo implementations)
    entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std
    entropy = entropy.sum(-1)

    value = critic.apply(params.critic_params, hidden).squeeze()
    return logprob, entropy, value

def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
    newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
    logratio = newlogprob - logp
    ratio = jnp.exp(logratio)
    approx_kl = ((ratio - 1) - logratio).mean()

    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

@jax.jit
def update_ppo(
    agent_state: TrainState,
    storage: Storage,
    key: jax.random.PRNGKey,
):
    def update_epoch(carry, unused_inp):
        agent_state, key = carry
        key, subkey = jax.random.split(key)

        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(subkey, x)
            x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
            return x

        flatten_storage = jax.tree_map(flatten, storage)
        shuffled_storage = jax.tree_map(convert_data, flatten_storage)

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

        agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_minibatch, agent_state, shuffled_storage
        )
        return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

    (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
        update_epoch, (agent_state, key), (), length=args.update_epochs
    )
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key