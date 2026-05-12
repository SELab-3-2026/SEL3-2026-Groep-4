from brittle_star_project.environment.env_config import MorphMode
import jax.numpy as jnp


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
