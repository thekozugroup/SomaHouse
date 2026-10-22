import numpy as np
import torch

from somahouse.brain import HouseBrain, graph, surgery
from somahouse.sim import observe, scenarios, transition
from somahouse.study import SEEDS, STRESS, TEST, VAL

torch.set_num_threads(2)


def test_rewire_preserves_in_out_degree_and_is_not_isomorphic_permutation():
    fly, _, _ = graph(32)
    rewired, _, info = graph(32, "rewired", 7)
    a, b = fly != 0, rewired != 0
    assert np.array_equal(a.sum(0), b.sum(0))
    assert np.array_equal(a.sum(1), b.sum(1))
    assert info["accepted_swaps"] > 0 and info["different_edges"] > 0
    assert not np.any(np.diag(b))


def test_duplicate_preserves_actions_and_has_private_weights():
    parent = HouseBrain()
    child = surgery(parent, "duplicate", 7)
    obs = torch.randn(20, 31)
    torch.testing.assert_close(child(obs), parent(obs), atol=1e-6, rtol=1e-5)
    assert child.stats()["modules"] == 8
    assert child.stats()["spoke_units"] == 256
    assert child.stats()["source_unique_ids"] == 32
    with torch.no_grad():
        child.input[4, 0, 0] += 3
    assert child.input[0, 0, 0] != child.input[4, 0, 0]


def test_cut_physically_removes_parameters_and_preserves_ids(tmp_path):
    parent = HouseBrain()
    child = surgery(parent, "cut", 7)
    assert child.stats()["spoke_units"] == 64
    assert child.stats()["allocated_parameters"] < parent.stats()["allocated_parameters"]
    assert set(n["body_id"] for n in child.nodes) <= set(n["body_id"] for n in parent.nodes)
    path = tmp_path / "brain.pt"; child.save(path)
    loaded = HouseBrain.load(path)
    obs = torch.randn(5, 31)
    torch.testing.assert_close(loaded(obs), child(obs), atol=0, rtol=0)


def test_no_hub_blocks_other_sensor_streams():
    brain = HouseBrain(hub=False)
    x = torch.randn(5, 31)
    y = x.clone(); y[:, 3:6] += 2
    # Thermal spoke sees no CO2. With no hub its heat action cannot change.
    torch.testing.assert_close(brain(x)[:, :3], brain(y)[:, :3], atol=0, rtol=0)
    assert not torch.equal(brain(x)[:, 3:6], brain(y)[:, 3:6])


def test_hub_transfers_other_sensor_stream_and_trainable_graph_receives_gradient():
    brain = HouseBrain()
    env = scenarios([157, 211], 4, training=True)
    state = env.initial
    x = observe(state, env, 0)
    y = x.clone(); y[:, 3:6] += 2
    assert not torch.equal(brain(x)[:, :3], brain(y)[:, :3])
    loss = 0
    for t in range(4):
        state, m = transition(state, brain(observe(state, env, t)), env, t)
        loss = loss + m["cost"].mean()
    loss.backward()
    for name in ("recurrent", "input", "hub_weight", "receive", "readout"):
        grad = getattr(brain, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    assert torch.count_nonzero(brain.recurrent.grad[~brain.edge_mask]) == 0


def test_checkpoint_exact_reload_and_lesions(tmp_path):
    b = HouseBrain()
    path = tmp_path / "brain.pt"; b.save(path)
    loaded = HouseBrain.load(path)
    obs = torch.randn(7, 31)
    torch.testing.assert_close(b(obs), loaded(obs), atol=0, rtol=0)
    assert not torch.equal(b(obs), b(obs, intervention="hub"))
    assert not torch.equal(b(obs), b(obs, intervention="recurrence"))


def test_registered_seed_sets_disjoint():
    assert len(SEEDS) == 3
    assert not (set(VAL) & set(TEST) or set(VAL) & set(STRESS) or set(TEST) & set(STRESS))
    assert min(VAL) >= 2_000_000
