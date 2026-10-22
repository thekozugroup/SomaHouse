import numpy as np
import pytest
import torch

from somahouse.sim import DT, FEATURES, State, actuate, baseline, observe, rollout, scenarios, transition


def test_passive_equilibrium_and_known_co2_decay():
    env = scenarios([101], 1)
    env.outside[:] = 22
    env.people.zero_(); env.daylight.zero_()
    state = State(torch.full((1, 3), 22.), torch.full((1, 3), 1000.), torch.zeros(1, 9))
    after, metrics = transition(state, torch.zeros(1, 9), env, 0)
    torch.testing.assert_close(after.temperature, state.temperature)
    torch.testing.assert_close(after.co2, torch.full((1, 3), 1000 - DT * .12 * 580))
    assert metrics["energy_kwh"].item() == 0


def test_heat_capacity_and_electrical_integration():
    env = scenarios([102], 1)
    env.outside[:] = 22; env.people.zero_(); env.daylight.zero_()
    state = State(torch.full((1, 3), 22.), torch.full((1, 3), 420.), torch.zeros(1, 9))
    action = torch.tensor([[.5, 0, 0, 0, 0, 0, 0, 0, 0.]])
    after, m = transition(state, action, env, 0)
    assert after.temperature[0, 0].item() == pytest.approx(22 + DT * 1.5 / env.capacity[0, 0], abs=2e-6)
    assert m["energy_kwh"].item() == pytest.approx(1.5 / 3.2 * DT)


def test_capacity_constraints_and_no_hidden_override():
    raw = torch.tensor([[2., -2., 1., -1., 2., .5, -2., 1., 3.]])
    a = actuate(raw)
    assert a[:, :3].abs().sum() <= 2.00001
    assert torch.all((a[:, 3:] >= 0) & (a[:, 3:] <= 1))
    reduced = actuate(raw, .45)
    assert reduced[:, :3].abs().sum() <= .90001
    assert reduced[:, 3:6].max() <= .45
    env = scenarios([103], 1)
    _, m = transition(env.initial, torch.zeros(1, 9), env, 0)
    assert m["executed"].count_nonzero() == 0


def test_scenarios_independent_of_batch_and_reproducible():
    single = scenarios([99], 24)
    batched = scenarios([42, 99], 24)
    torch.testing.assert_close(single.outside[:, 0], batched.outside[:, 1])
    torch.testing.assert_close(single.people[:, 0], batched.people[:, 1])
    torch.testing.assert_close(single.initial.temperature[0], batched.initial.temperature[1])
    assert observe(single.initial, single, 0).shape == (1, FEATURES)


def test_action_derivative_matches_finite_difference():
    env = scenarios([107], 1)
    a = torch.full((1, 9), .3, requires_grad=True)
    state, _ = transition(env.initial, a, env, 0)
    value = state.temperature[0, 0] + state.co2[0, 0] * .01
    analytical, = torch.autograd.grad(value, a)
    epsilon = .002
    for index in (0, 3):
        values = []
        for sign in (-1, 1):
            candidate = a.detach().clone(); candidate[0, index] += sign * epsilon
            s, _ = transition(env.initial, candidate, env, 0)
            values.append(float(s.temperature[0, 0] + s.co2[0, 0] * .01))
        numerical = (values[1] - values[0]) / (2 * epsilon)
        assert analytical[0, index].item() == pytest.approx(numerical, abs=.003)


@pytest.mark.parametrize("kind", ["feedback", "balance", "off"])
def test_full_48_hour_baselines_finite(kind):
    with torch.no_grad():
        metrics, trace = rollout(kind, scenarios([123, 345], 288), trace=True)
    assert len(trace) == 288
    assert all(torch.isfinite(v).all() for v in metrics.values())
    assert metrics["occupied_zone_hours"].min() > 0
    assert metrics["energy_kwh"].min() >= 0
    assert all(350 < c < 5000 for step in trace for c in step["co2"])


def test_sensor_fault_changes_observation_not_true_physics():
    a = scenarios([42], 1)
    b = scenarios([42], 1, stress="temperature_bias")
    torch.testing.assert_close(a.initial.temperature, b.initial.temperature)
    torch.testing.assert_close(observe(b.initial, b, 0)[:, :3] - observe(a.initial, a, 0)[:, :3], torch.full((1, 3), .25))
    dropout = scenarios([42], 1, stress="co2_dropout")
    torch.testing.assert_close(observe(dropout.initial, dropout, 0)[:, 3:6], torch.full((1, 3), .23))
