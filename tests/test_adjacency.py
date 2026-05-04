import jax.numpy as jnp
import numpy as np

from brittle_star_project.MLPs import build_adjacency
from brittle_star_project.environment.env_config import MorphMode


def assert_symmetric(adj):
    assert jnp.all(adj == adj.T)


def test_centralized():
    adj = build_adjacency([4, 4, 4, 4, 4], MorphMode.CENTRALIZED)

    assert adj.shape == (1, 1)
    assert adj[0, 0] == 1


def test_fully_connected():
    adj = build_adjacency([4, 4, 4, 4, 4], MorphMode.FULLY_CONNECTED)

    assert adj.shape == (5, 5)
    assert jnp.all(adj == 1)


def test_ring():
    adj = build_adjacency([4, 4, 4, 4, 4], MorphMode.RING)

    assert adj.shape == (5, 5)
    assert_symmetric(adj)

    # each node should connect to itself + 2 neighbors
    for i in range(5):
        assert adj[i, i] == 1
        assert jnp.sum(adj[i]) == 3


def test_segment_structure():
    segments = [4, 4, 4, 4, 4]
    adj = build_adjacency(segments, MorphMode.SEGMENT)

    num_arms = 5
    num_segments = sum(segments)
    num_nodes = num_arms + num_segments

    assert adj.shape == (num_nodes, num_nodes)

    # --- ring connectivity ---
    for i in range(num_arms):
        assert adj[i, i] == 1
        assert adj[i, (i - 1) % num_arms] == 1
        assert adj[i, (i + 1) % num_arms] == 1

    # --- segment chain checks ---
    offset = num_arms
    for arm in range(5):
        for i in range(4):
            node = offset + arm * 4 + i

            # self
            assert adj[node, node] == 1

            # chain neighbors
            if i > 0:
                assert adj[node, node - 1] == 1
            if i < 3:
                assert adj[node, node + 1] == 1

    # --- ring ↔ segment connections ---
    for arm in range(5):
        first_seg = num_arms + arm * 4
        assert adj[arm, first_seg] == 1
        assert adj[first_seg, arm] == 1

    save_adj(adj)


def save_adj(adj, name="adjacency_debug.txt"):
    a = np.array(adj)

    with open(name, "w") as f:
        f.write("\nAdjacency matrix:\n")
        f.write("   " + " ".join([f"{i:2d}" for i in range(a.shape[0])]) + "\n")

        for i, row in enumerate(a):
            line = f"{i:2d}  " + "  ".join(["█" if x > 0 else "." for x in row])
            f.write(line + "\n")


def test_no_isolated_nodes():
    adj = build_adjacency([4, 4, 4, 4, 4], MorphMode.SEGMENT)

    # no node should be completely isolated
    assert jnp.all(jnp.sum(adj, axis=0) > 0)
