from functools import partial

import flax
import jax
import jax.numpy as jnp


# Chose to use a class as it seemed the easiest way to integrate the CleanRL code style
# with our need to seperate concerns
class PPO:
    def __init__(
        self, args, input_network, action_network, critic, critic_network, message_passer=None
    ):
        self.args = args

        if not message_passer:
            message_passer = identity

        self.ppo_loss_grad_fn = jax.value_and_grad(
            partial(
                ppo_loss,
                args=args,
                input_network_apply=input_network.apply,
                action_network_apply=action_network.apply,
                critic_apply=critic.apply,
                critic_network_apply=critic_network.apply,
                message_passer=message_passer,
            ),
            has_aux=True,
        )

    # This PPO class should be initialized only once,
    # or this function will need to recompile
    @partial(jax.jit, static_argnums=0)
    def update_ppo(self, agent_state, storage, key):
        args = self.args
        ppo_loss_grad_fn = self.ppo_loss_grad_fn

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
                return agent_state, (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    grads,
                )

            agent_state, metrics = jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
            return (agent_state, key), metrics

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


"""
Should be ok to use partial here, since the references to network,
actor and critic should not change at runtime
The cost of seperating concerns is to somehow pass these values
that are now not in the same scope
"""


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def get_action_and_value2(
    input_apply,
    action_apply,
    message_passer,
    critic_apply,
    critic_network_apply,
    params: flax.core.FrozenDict,
    x: jnp.ndarray,
    action: jnp.ndarray,
):
    hidden_network = input_apply(params["network_params"], x)
    hidden_critic = critic_network_apply(params["critic_network_params"], x)
    hidden_network = message_passer(hidden_network)
    mean, log_std = action_apply(params["actor_params"], hidden_network)
    std = jnp.exp(log_std)

    logprob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)).sum(-1)
    entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).sum(-1)
    value = critic_apply(params["critic_params"], hidden_critic).squeeze(-1)

    return logprob, entropy, value


def ppo_loss(
    params,
    x,
    a,
    logp,
    mb_advantages,
    mb_returns,
    args,
    input_network_apply,
    action_network_apply,
    message_passer,
    critic_apply,
    critic_network_apply,
):
    newlogprob, entropy, newvalue = get_action_and_value2(
        input_network_apply,
        action_network_apply,
        message_passer,
        critic_apply,
        critic_network_apply,
        params,
        x,
        a,
    )
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


def identity(hidden):
    """
    Used for seamless jax integration,
    avoids having branching inside jitted function,
    used as message_passer in case it is not given,
    (in case of centralized lvl)
    """

    return hidden
