import jax
import jax.numpy as jnp


def message_passer(params, hidden, adjacency):
    # take current hidden state per env, per agent, needs to communicate according to adjacency
    # adjacency can be assumed to be the same per env? or mix it too idk
    # rest is simple, use adjacency to make which hidden states we can combine
    # repeat X times with the new combined vectors
    # return result..
    # use jax lax scan stuff or vmap for speed
    def per_env(h):
        # compute messages per agent
        def compute_messages(h_i, h_all):
            # broadcast h_i with all neighbors
            h_i_rep = jnp.repeat(h_i[None, :], h_all.shape[0], axis=0)
            msg_input = jnp.concatenate([h_i_rep, h_all], axis=-1)
            messages = messager_apply(params, msg_input)
            return messages

        msgs = jax.vmap(compute_messages, in_axes=(0, None))(h, h)
        msgs = msgs * adjacency[..., None]  # mask neighbors
        agg = msgs.sum(axis=1)
        return agg

    return jax.vmap(per_env)(hidden)
