"""Read back artifacts and independently recompute recorded physical trajectories."""
from __future__ import annotations

import csv
import argparse
import json
import tempfile

import numpy as np
import torch

from .brain import ROOT, HouseBrain
from .sim import DT, observe, rollout, scenarios
from .study import REPORTS, SEEDS, TEST, sha, write_json


def numpy_replay(trace_data):
    """Float64 NumPy equations independent of Torch transition(), using executed actions."""
    env = scenarios([trace_data["scenario_seed"]], 288)
    temp = env.initial.temperature[0].numpy().astype(np.float64)
    co2 = env.initial.co2[0].numpy().astype(np.float64)
    cap = env.capacity[0].numpy().astype(np.float64)
    leak = env.conductance[0].numpy().astype(np.float64)
    volume = env.volume[0].numpy().astype(np.float64)
    sums = {"energy_kwh": 0., "thermal_degree_hours": 0., "co2_ppm_hours": 0., "light_lux_hours": 0.}
    errors = {"temperature": 0., "co2": 0., "lux": 0., "power_kw": 0.}
    for i, row in enumerate(trace_data["trace"]):
        a = np.asarray(row["action"], np.float64)
        outside = float(env.outside[i, 0])
        people = env.people[i, 0].numpy().astype(np.float64)
        day = env.daylight[i, 0].numpy().astype(np.float64)
        target = env.setpoint[i, 0].numpy().astype(np.float64)
        ach = .12 + 3.4 * a[3:6]
        q = leak * (outside - temp) + .000335 * volume * ach * (outside - temp)
        q += .025 * (temp.mean() - temp) + .12 * people + .00035 * day + .07 * a[6:9] + 3 * a[:3]
        temp = temp + DT * q / cap
        co2 = co2 + DT * (18000 / volume * people - ach * (co2 - 420))
        lux = day + 450 * a[6:9]
        power = np.sum(3 * np.abs(a[:3]) / 3.2 + .045 * a[3:6] ** 3 + .07 * a[6:9])
        for key, value in (("temperature", temp), ("co2", co2), ("lux", lux), ("power_kw", power)):
            errors[key] = max(errors[key], float(np.max(np.abs(value - row[key]))))
        occupied = people > 0
        sums["energy_kwh"] += power * DT
        sums["thermal_degree_hours"] += np.sum(np.maximum(np.abs(temp - target) - 1, 0) * occupied) * DT / 3
        sums["co2_ppm_hours"] += np.sum(np.maximum(co2 - 1000, 0) * occupied) * DT / 3
        sums["light_lux_hours"] += np.sum(np.maximum(300 - lux, 0) * occupied) * DT / 3
    assert errors["temperature"] < .002, errors
    assert errors["co2"] < .05 and errors["lux"] < .002 and errors["power_kw"] < .0001, errors
    for key, value in sums.items():
        assert abs(value - trace_data["metrics"][key]) < max(.01, abs(value) * 2e-4), (key, value, trace_data["metrics"][key])
    return {"controller": trace_data["controller"], "steps": len(trace_data["trace"]),
            "max_absolute_errors": errors, "independent_totals": sums}


@torch.no_grad()
def verify(*, portable=False):
    torch.set_num_threads(2)
    selection_path = REPORTS / "selection.json"
    selection = json.loads(selection_path.read_text())
    results = json.loads((REPORTS / "results.json").read_text())
    assert results["selection_sha256"] == sha(selection_path)
    assert selection["protocol_sha256"] == sha(ROOT / "research/protocol.md")
    provenance = json.loads((ROOT / "data/connectome/provenance.json").read_text())
    assert provenance["seed_sha256"] == sha(ROOT / "data/connectome/seed.npz")
    model_checks = []
    obs = observe(scenarios([731, 972], 1).initial, scenarios([731, 972], 1), 0)
    with tempfile.TemporaryDirectory() as folder:
        for run in selection["runs"]:
            path = ROOT / run["selected_checkpoint"]
            initial_path = path.with_name(path.stem + "-initial.pt")
            assert sha(path) == run["sha256"] and sha(initial_path) == run["initial_sha256"]
            brain = HouseBrain.load(path)
            initial = HouseBrain.load(initial_path)
            check_path = f"{folder}/reload.pt"; brain.save(check_path)
            loaded = HouseBrain.load(check_path)
            assert torch.equal(brain(obs), loaded(obs))
            assert all(torch.isfinite(v).all() for v in brain.state_dict().values())
            changed = int(torch.count_nonzero((brain.recurrent - initial.recurrent) * brain.edge_mask))
            assert changed > 0 or run["selected_update"] == 0
            model_checks.append({"id": run["id"], "selected_sha256": sha(path),
                                 "initial_sha256": sha(initial_path), "exact_action_reload": True,
                                 "changed_active_recurrent_weights": changed})
    rows = list(csv.DictReader((REPORTS / "episodes.csv").open()))
    assert len(rows) == results["episodes"]
    replay = []
    for seed in SEEDS:
        brain = HouseBrain.load(ROOT / "checkpoints" / f"{results['primary']}-s{seed}.pt")
        metrics, _ = rollout(brain, scenarios(TEST, 288))
        recorded = {int(r["scenario_seed"]): r for r in rows if r["controller"] == results["primary"] and int(r["training_seed"]) == seed and r["stress"] == "normal"}
        maximum = 0.
        for i, scenario_seed in enumerate(TEST):
            for key, values in metrics.items():
                delta = abs(float(values[i]) - float(recorded[scenario_seed][key]))
                maximum = max(maximum, delta)
                tolerance = 1e-7
                if portable:
                    # BLAS/CPU instruction sets vary. These are absolute unit-aware
                    # tolerances, not permission to change the stored checkpoints.
                    tolerance = {"cost": 2e-5, "energy_kwh": .001, "thermal_degree_hours": .001,
                                 "co2_ppm_hours": .15, "light_lux_hours": .05,
                                 "occupied_zone_hours": 1e-5, "action_movement": .001,
                                 "peak_kw": .0001}.get(key, .17)
                assert delta < tolerance, (seed, scenario_seed, key, delta, tolerance)
        replay.append({"seed": seed, "episodes": len(TEST), "max_metric_error": maximum,
                       "mode": "portable_unit_tolerances" if portable else "strict_1e-7"})
    physics = [numpy_replay(json.loads(p.read_text())) for p in sorted(REPORTS.glob("trace-*.json"))]
    report = {"passed": True, "checkpoint_files": len(model_checks) * 2, "model_checks": model_checks,
              "primary_test_replay": replay, "independent_numpy_physics": physics,
              "source_seed_sha256": sha(ROOT / "data/connectome/seed.npz"),
              "selection_sha256": sha(selection_path), "protocol_sha256": sha(ROOT / "research/protocol.md"),
              "test_rows": len(rows), "weights_only_loader": True}
    write_json(REPORTS / "verification.json", report)
    print(json.dumps({"passed": True, "checkpoint_files": report["checkpoint_files"],
                      "replayed_episodes": sum(r["episodes"] for r in replay),
                      "numpy_trace_steps": sum(r["steps"] for r in physics)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", action="store_true", help="Allow unit-aware CPU/BLAS rounding differences")
    verify(portable=parser.parse_args().portable)
