from functools import partial

import jax
import jax.numpy as jnp
from jax import debug
from flax.core import FrozenDict
from experiment_logger import get_logger
from brittle_star_project.utils import logged_jit

logger = get_logger()


# Chose to use a class as it seemed the easiest way to integrate the CleanRL code style
# with our need to seperate concerns
class PPO:
    def __init__(
        self,
        args,
        sensor_apply,
        actor_apply,
        critic_apply,
        feature_extractor_apply,
        message_passer=None,
    ):
        self.args = args

        if not message_passer:
            message_passer = identity

        self.ppo_loss_grad_fn = jax.value_and_grad(
            partial(
                ppo_loss,
                args=args,
                sensor_apply=sensor_apply,
                actor_apply=actor_apply,
                critic_apply=critic_apply,
                feature_extractor_apply=feature_extractor_apply,
                message_passer=message_passer,
            ),
            has_aux=True,
        )

    # This PPO class should be initialized only once,
    # or this function will need to recompile
    @partial(logged_jit, static_argnums=0)
    def update_ppo(self, agent_state, storage, key):
        debug.callback(logger.debug, f"[PPO] storage.obs shape: {storage.obs.shape}")
        debug.callback(logger.debug, f"[PPO] storage.actions shape: {storage.actions.shape}")
        debug.callback(logger.debug, f"[PPO] storage.logprobs shape: {storage.logprobs.shape}")
        debug.callback(logger.debug, f"[PPO] storage.advantages shape: {storage.advantages.shape}")
        debug.callback(logger.debug, f"[PPO] storage.returns shape: {storage.returns.shape}")

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
                debug.callback(logger.debug, f"[PPO] minibatch.obs: {minibatch.obs.shape}")
                debug.callback(logger.debug, f"[PPO] minibatch.actions: {minibatch.actions.shape}")
                debug.callback(
                    logger.debug, f"[PPO] minibatch.logprobs: {minibatch.logprobs.shape}"
                )
                debug.callback(
                    logger.debug, f"[PPO] minibatch.advantages: {minibatch.advantages.shape}"
                )
                debug.callback(logger.debug, f"[PPO] minibatch.returns: {minibatch.returns.shape}")

                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, metrics = jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
            return (agent_state, key), metrics

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


"""
Should be ok to use partial here, since the references to network,
actor and critic should not change at runtime
The cost of seperating concerns is to somehow pass these values
that are now not in the same scope
"""


@partial(logged_jit, static_argnums=(0, 1, 2, 3, 4))
def get_action_and_value(
    sensor_apply,
    actor_apply,
    message_passer,
    critic_apply,
    feature_extractor_apply,
    params: FrozenDict,
    x: jnp.ndarray,
    action: jnp.ndarray,
):
    hidden_sensor = sensor_apply(params["sensor_params"], x)
    hidden_critic = feature_extractor_apply(params["feature_extractor_params"], x)
    hidden_sensor = message_passer(params["message_passer_params"], hidden_sensor)

    debug.callback(logger.debug, f"[SHAPE] hidden_sensor: {hidden_sensor.shape}")
    debug.callback(logger.debug, f"[SHAPE] hidden_critic: {hidden_critic.shape}")

    mean, log_std = actor_apply(params["actor_params"], hidden_sensor)

    debug.callback(logger.debug, f"[SHAPE] mean: {mean.shape}")
    debug.callback(logger.debug, f"[SHAPE] log_std: {log_std.shape}")
    debug.callback(logger.debug, f"[SHAPE] action: {action.shape}")

    log_std = jnp.clip(log_std, -5, 2)
    std = jnp.exp(log_std)

    logprob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    debug.callback(logger.debug, f"[SHAPE] logprob pre-sum: {logprob.shape}")

    logprob = logprob.sum(axis=(-2, -1))
    debug.callback(logger.debug, f"[SHAPE] logprob final: {logprob.shape}")

    entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_std).sum(axis=(-2, -1))
    value = critic_apply(params["critic_params"], hidden_critic).squeeze(-1)
    debug.callback(logger.debug, f"[SHAPE] value: {value.shape}")

    return logprob, entropy, value


def ppo_loss(
    params,
    x,
    a,
    logp,
    mb_advantages,
    mb_returns,
    args,
    sensor_apply,
    actor_apply,
    message_passer,
    critic_apply,
    feature_extractor_apply,
):
    newlogprob, entropy, newvalue = get_action_and_value(
        sensor_apply,
        actor_apply,
        message_passer,
        critic_apply,
        feature_extractor_apply,
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
