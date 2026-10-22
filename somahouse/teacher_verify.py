"""Independent readback for the second registered study; never retrains."""
import argparse
import csv
import json
import tempfile

import torch

from .brain import ROOT, HouseBrain
from .sim import actuate, observe, rollout, scenarios, transition
from .study import sha, write_json
from .teacher import CKPT, OUT, PRIMARY, PROTOCOL, TEST, VAL, Teacher, paired
from .verify import numpy_replay


@torch.no_grad()
def verify(portable=False):
    torch.set_num_threads(2)
    selection = json.loads((OUT / "selection.json").read_text())
    result = json.loads((OUT / "results.json").read_text())
    assert result["selection_sha256"] == sha(OUT / "selection.json")
    assert selection["protocol_sha256"] == sha(PROTOCOL)
    assert selection["teacher_sha256"] == sha(OUT / "teacher.json")
    rows = list(csv.DictReader((OUT / "episodes.csv").open()))
    numeric = [{k: (float(v) if k not in ("controller", "split", "stress") else v) for k, v in r.items()} for r in rows]
    assert len(rows) == result["episodes"]
    for comparison in result["comparisons"]:
        assert paired(numeric, comparison["a"], comparison["b"], comparison["metric"]) == comparison
    obs = observe(scenarios([5_123_456], 1).initial, scenarios([5_123_456], 1), 0)
    checked, policies = [], [("teacher", 0, Teacher(selection["teacher"]["config"])), ("balance", 0, "balance")]
    with tempfile.TemporaryDirectory() as folder:
        for run in selection["runs"]:
            path = ROOT / run["checkpoint"]
            assert sha(path) == run["sha256"]
            brain = HouseBrain.load(path)
            brain.save(f"{folder}/reload.pt")
            assert torch.equal(brain(obs), HouseBrain.load(f"{folder}/reload.pt")(obs))
            assert all(torch.isfinite(v).all() for v in brain.state_dict().values())
            initial = (ROOT / run["initial_checkpoint"]) if run["stage"] == "imitation" else (ROOT / run["parent_checkpoint"])
            if run["stage"] == "imitation": assert sha(initial) == run["initial_sha256"]
            parent = HouseBrain.load(initial)
            changed = int(torch.count_nonzero((brain.recurrent - parent.recurrent) * brain.edge_mask))
            assert changed > 0 or run["selected_update"] == 0
            checked.append({"id": run["id"], "sha256": sha(path), "exact_reload": True,
                            "active_recurrent_weights_changed": changed})
            policies.append((run["family"], run["seed"], brain))
    replay = []
    for name, seed, policy in policies:
        metrics, _ = rollout(policy, scenarios(TEST, 288))
        recorded = {int(r["scenario_seed"]): r for r in rows if r["controller"] == name and int(r["training_seed"]) == seed and r["stress"] == "normal"}
        maximum = 0.
        for i, scenario_seed in enumerate(TEST):
            for key, value in metrics.items():
                delta = abs(float(value[i]) - float(recorded[scenario_seed][key])); maximum = max(maximum, delta)
                tolerance = 1e-7 if not portable else {"cost": 2e-5, "energy_kwh": .001,
                     "thermal_degree_hours": .001, "co2_ppm_hours": .15, "light_lux_hours": .05,
                     "occupied_zone_hours": 1e-5, "action_movement": .001, "peak_kw": .0001}.get(key, .17)
                assert delta < tolerance, (name, seed, key, delta, tolerance)
        replay.append({"controller": name, "training_seed": seed, "episodes": len(TEST), "max_metric_error": maximum})
    physics = [numpy_replay(json.loads(p.read_text())) for p in sorted(OUT.glob("trace-*.json"))]
    # An interpretable supervision diagnostic, separate from scored closed-loop cost.
    teacher = Teacher(selection["teacher"]["config"])
    env = scenarios(VAL, 288); state = env.initial
    action_error = {r["id"]: 0. for r in selection["runs"]}
    models = {r["id"]: HouseBrain.load(ROOT / r["checkpoint"]) for r in selection["runs"]}
    for t in range(env.steps):
        obs = observe(state, env, t); target = actuate(teacher(obs))
        for name, model in models.items():
            action_error[name] += float(((actuate(model(obs)) - target) ** 2).mean()) / env.steps
        state, _ = transition(state, teacher(obs), env, t)
    report = {"passed": True, "mode": "portable" if portable else "strict_1e-7",
              "selected_checkpoints": len(checked), "initial_checkpoints": len(list(CKPT.glob("*-initial-*.pt"))),
              "models": checked, "normal_test_replay": replay, "numpy_physics": physics,
              "selection_sha256": sha(OUT / "selection.json"), "protocol_sha256": sha(PROTOCOL),
              "test_rows": len(rows), "comparisons_recomputed": len(result["comparisons"]),
              "validation_teacher_state_action_mse": action_error,
              "mse_diagnostic": "unweighted executed-action MSE on 32 validation teacher trajectories; not a checkpoint selection rule"}
    write_json(OUT / "verification.json", report)
    print(json.dumps({"passed": True, "selected_models": len(checked), "replayed_episodes": sum(r["episodes"] for r in replay)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--portable", action="store_true")
    verify(parser.parse_args().portable)
