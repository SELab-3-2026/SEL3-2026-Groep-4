import jax
import jax.numpy as jnp
from brittle_star_project.environment.padded_obs_wrapper import (
    compute_padding_masks,
    pad_observations_batched,
)

# We use Actor and OneDenseLayerMLP (as the critic) based on your mlps.py
from brittle_star_project.MLPs.mlps import Actor, OneDenseLayerMLP


def test_centralized_forward_pass_with_padding():
    batch_size = 2

    # 1. Simulate Amputated Observation [4, 0, 4, 2, 4] -> 14 segments total
    # 14 segments * 2 = 28 joints
    amputated_obs = {
        "joint_position": jnp.zeros((batch_size, 28)),
        "joint_velocity": jnp.zeros((batch_size, 28)),
        "segment_contact": jnp.zeros((batch_size, 14)),
    }

    # 2. Pad Observation using the boolean scattering wrapper
    masks = compute_padding_masks(segments_per_arm=(4, 0, 4, 2, 4))
    padded_obs = pad_observations_batched(amputated_obs, masks)

    # Assertions to ensure padding sizes are correct (40 joints, 20 segments)
    assert padded_obs["joint_position"].shape == (batch_size, 40), "Padding failed for joint keys"
    assert padded_obs["segment_contact"].shape == (batch_size, 20), (
        "Padding failed for segment keys"
    )

    # 3. Concatenate for Centralized MLP (simulating the global state vector)
    global_state = jnp.concatenate(
        [padded_obs["joint_position"], padded_obs["joint_velocity"], padded_obs["segment_contact"]],
        axis=-1,
    )

    # 40 + 40 + 20 = 100 dimensions
    assert global_state.shape == (batch_size, 100), (
        f"Expected global state shape (2, 100), got {global_state.shape}"
    )

    # 4. Initialize dummy networks (40 actuators for the max morphology output)
    actor = Actor(action_dim=40)
    critic = OneDenseLayerMLP()  # Acts as the centralized critic

    rng = jax.random.PRNGKey(0)
    rng_a, rng_c = jax.random.split(rng)

    # Initialize Flax variables
    actor_params = actor.init(rng_a, global_state)
    critic_params = critic.init(rng_c, global_state)

    # 5. Forward Pass Assertions
    action_mean, action_log_std = actor.apply(actor_params, global_state)
    value = critic.apply(critic_params, global_state)

    assert action_mean.shape == (batch_size, 40), f"Actor mean shape mismatch: {action_mean.shape}"
    assert action_log_std.shape == (40,), f"Actor log_std shape mismatch: {action_log_std.shape}"
    assert value.shape == (batch_size, 1) or value.shape == (batch_size,), (
        f"Critic value shape mismatch: {value.shape}"
    )


if __name__ == "__main__":
    test_centralized_forward_pass_with_padding()
