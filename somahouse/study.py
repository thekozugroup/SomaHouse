"""Frozen three-seed study. Train/validate first; seal selection before test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .brain import ROOT, HouseBrain, surgery
from .sim import COST_WEIGHTS, DT, observe, rollout, scenarios, transition

SEEDS = [7, 19, 41]
VAL = list(range(2_000_001, 2_000_025))
TEST = list(range(3_000_001, 3_000_061))
STRESS = list(range(4_000_001, 4_000_017))
SPECS = {"fly32": (32, "fly", True), "fly64": (64, "fly", True),
         "rewired32": (32, "rewired", True), "dense32": (32, "dense", True),
         "nohub32": (32, "fly", False)}
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def validate(brain):
    metrics, _ = rollout(brain, scenarios(VAL, 144))
    return {k: float(v.mean()) for k, v in metrics.items()}


def train_one(name, seed, brain, updates, *, descendant=False):
    ident = f"{name}-s{seed}"
    selected = CHECKPOINTS / f"{ident}.pt"
    info_path = REPORTS / "training" / f"{ident}.json"
    if info_path.exists():
        data = json.loads(info_path.read_text())
        if data.get("completed_updates") == updates and selected.exists() and sha(selected) == data["sha256"]:
            print(f"skip verified {ident}", flush=True)
            return data
        raise RuntimeError(f"Incomplete or changed run {ident}; preserve it before rerunning")
    initial_path = CHECKPOINTS / f"{ident}-initial.pt"
    brain.save(initial_path)
    v = validate(brain)
    best = v["cost"]
    best_update = 0
    brain.save(selected)
    curve = [{"update": 0, "validation": v}]
    if descendant and name == "duplicate":
        # Break copy symmetry only after saving the exactly preserving initial checkpoint.
        generator = torch.Generator().manual_seed(seed + 988)
        with torch.no_grad():
            noise = torch.randn(brain.input.shape, generator=generator) * .002
            brain.input.add_(noise)
    optimizer = torch.optim.Adam(brain.parameters(), lr=.003 if not descendant else .0015)
    rng = np.random.default_rng(seed + (500 if descendant else 0))
    start = time.perf_counter()
    norms = []
    for update in range(1, updates + 1):
        episode_seeds = rng.integers(1000, 2_000_000, 24).tolist()
        env = scenarios(episode_seeds, 24, training=True)
        state = env.initial
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros(())
        for t in range(24):
            action = brain(observe(state, env, t))
            state, metrics = transition(state, action, env, t)
            loss = loss + metrics["cost"].mean() / 24
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss {ident} {update}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(brain.parameters(), 5.0, error_if_nonfinite=True)
        norms.append(float(norm))
        optimizer.step()
        if update % 50 == 0 or update == updates:
            v = validate(brain)
            curve.append({"update": update, "training_cost": float(loss.detach()), "validation": v,
                          "seconds": time.perf_counter() - start})
            if v["cost"] < best:
                best, best_update = v["cost"], update
                brain.save(selected)
            print(f"{ident} {update}/{updates} val={v['cost']:.4f} best={best:.4f}@{best_update}", flush=True)
    data = {"id": ident, "family": name, "seed": seed, "completed_updates": updates,
            "selected_update": best_update, "validation_cost": best,
            "training_seconds": time.perf_counter() - start, "curve": curve,
            "sha256": sha(selected), "initial_sha256": sha(initial_path),
            "stats": HouseBrain.load(selected).stats(), "gradient_norm_max": max(norms),
            "training_windows": updates * 24, "training_transitions": updates * 24 * 24,
            "selected_checkpoint": str(selected.relative_to(ROOT))}
    write_json(info_path, data)
    return data


def train_study():
    torch.set_num_threads(2)
    CHECKPOINTS.mkdir(exist_ok=True)
    (REPORTS / "training").mkdir(parents=True, exist_ok=True)
    if (REPORTS / "results.json").exists():
        raise RuntimeError("Final test already opened; start a new registered study instead of tuning")
    runs = []
    for seed in SEEDS:
        for name, (width, topology, hub) in SPECS.items():
            runs.append(train_one(name, seed, HouseBrain(width, seed, topology, hub), 300))
        parent = HouseBrain.load(CHECKPOINTS / f"fly32-s{seed}.pt")
        for operation in ("continue", "cut", "duplicate"):
            runs.append(train_one(operation, seed, surgery(parent, operation, seed), 150, descendant=True))
    families = {}
    for r in runs:
        families.setdefault(r["family"], []).append(r)
    candidates = [k for k in ("fly32", "fly64", "continue", "cut", "duplicate")]
    costs = {k: np.mean([r["validation_cost"] for r in families[k]]) for k in candidates}
    best = min(costs.values())
    eligible = [k for k in candidates if costs[k] <= best * 1.02]
    primary = min(eligible, key=lambda k: (families[k][0]["stats"]["spoke_units"], costs[k]))
    # All three matched seed improvements are required before considering added copies.
    duplicate_improves_all = all(next(r for r in families["duplicate"] if r["seed"] == s)["validation_cost"]
                                 < next(r for r in families["continue"] if r["seed"] == s)["validation_cost"] for s in SEEDS)
    if primary == "duplicate" and not duplicate_improves_all:
        primary = min([k for k in candidates if k != "duplicate"], key=lambda k: (costs[k], families[k][0]["stats"]["spoke_units"]))
    selection = {"primary_family": primary, "family_validation_costs": costs,
                 "duplicate_improves_all_validation_seeds": duplicate_improves_all,
                 "selected_before_test": True, "training_seeds": SEEDS,
                 "validation_seeds": VAL, "test_seeds": TEST, "stress_seeds": STRESS,
                 "protocol_sha256": sha(ROOT / "research/protocol.md"),
                 "cost_weights": COST_WEIGHTS, "runs": runs,
                 "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                 "platform": platform.platform(), "threads": 2}}
    write_json(REPORTS / "selection.json", selection)
    print(f"SEALED PRIMARY: {primary}; validation means {costs}", flush=True)


def rows_for(name, seed, metrics, env, split):
    rows = []
    for i, scenario_seed in enumerate(env.seeds):
        row = {"controller": name, "training_seed": seed, "scenario_seed": scenario_seed,
               "split": split, "stress": env.stress}
        row.update({k: float(v[i]) for k, v in metrics.items()})
        occupied = max(row["occupied_zone_hours"], 1e-8)
        for key in ("thermal", "co2", "light"):
            row[f"{key}_compliance"] = row[f"{key}_compliant_hours"] / occupied
        rows.append(row)
    return rows


def bootstrap_difference(rows, a, b, key="cost", stress="normal"):
    def group(name):
        grouped = {}
        for r in rows:
            if r["controller"] == name and r["stress"] == stress:
                grouped.setdefault(r["scenario_seed"], []).append(r[key])
        return {k: np.mean(v) for k, v in grouped.items()}
    left, right = group(a), group(b)
    keys = sorted(left.keys() & right.keys())
    delta = np.array([left[k] - right[k] for k in keys])
    rng = np.random.default_rng(918)
    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
    return {"a": a, "b": b, "metric": key, "difference": float(delta.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "paired_scenarios": len(keys),
            "negative_favors": a}


@torch.no_grad()
def evaluate_study():
    torch.set_num_threads(2)
    selection = json.loads((REPORTS / "selection.json").read_text())
    if (REPORTS / "results.json").exists():
        raise RuntimeError("Evaluation exists; use verify to replay without replacing it")
    if selection["protocol_sha256"] != sha(ROOT / "research/protocol.md"):
        raise RuntimeError("Protocol changed after selection")
    selection_sha = sha(REPORTS / "selection.json")
    rows, timings = [], []
    primary = selection["primary_family"]
    splits = [("test", "normal", TEST)] + [("stress", s, STRESS) for s in ("weather", "temperature_bias", "co2_dropout", "capacity")]
    policies = [(name, 0, name) for name in ("off", "feedback", "balance")]
    for run in selection["runs"]:
        path = ROOT / run["selected_checkpoint"]
        if sha(path) != run["sha256"]:
            raise RuntimeError("Changed checkpoint")
        policies.append((run["family"], run["seed"], HouseBrain.load(path)))
    for seed in SEEDS:
        policies.append(("untrained32", seed, HouseBrain.load(CHECKPOINTS / f"fly32-s{seed}-initial.pt")))
    start = time.perf_counter()
    for name, seed, policy in policies:
        for split, stress, episode_seeds in splits:
            env = scenarios(episode_seeds, 288, stress=stress)
            metrics, _ = rollout(policy, env)
            rows.extend(rows_for(name, seed, metrics, env, split))
        obs = observe(scenarios([TEST[0]], 1).initial, scenarios([TEST[0]], 1), 0)
        f = (lambda: __import__("somahouse.sim", fromlist=["baseline"]).baseline(obs, policy)) if isinstance(policy, str) else lambda: policy(obs)
        for _ in range(20):
            f()
        samples = []
        for _ in range(200):
            t = time.perf_counter_ns(); f(); samples.append((time.perf_counter_ns() - t) / 1e6)
        timings.append({"controller": name, "training_seed": seed, "median_ms": float(np.median(samples)),
                        "p95_ms": float(np.quantile(samples, .95)), "batch": 1, "samples": 200})
        print(f"evaluated {name} seed={seed}", flush=True)
    with (REPORTS / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summaries = []
    for name in sorted({r["controller"] for r in rows}):
        for stress in ("normal", "weather", "temperature_bias", "co2_dropout", "capacity"):
            subset = [r for r in rows if r["controller"] == name and r["stress"] == stress]
            keys = [k for k, v in subset[0].items() if isinstance(v, float)]
            summaries.append({"controller": name, "stress": stress, "episodes": len(subset),
                              **{k: float(np.mean([r[k] for r in subset])) for k in keys}})
    comparisons = [bootstrap_difference(rows, a, b) for a, b in
                   [(primary, "feedback"), (primary, "balance"), ("duplicate", "continue"),
                    ("cut", "continue"), ("fly32", "rewired32"), ("fly32", "dense32"),
                    ("fly32", "nohub32"), ("fly32", "untrained32")]]
    interventions = []
    for seed in SEEDS:
        brain = HouseBrain.load(CHECKPOINTS / f"{primary}-s{seed}.pt")
        for lesion in (None, "hub", "thermal", "air", "light", "context", "recurrence") + tuple(f"class:{c}" for c in sorted({n["class"] or "Unclassified" for n in brain.nodes})):
            metrics, _ = rollout(brain, scenarios(VAL, 144), intervention=lesion)
            interventions.append({"seed": seed, "lesion": lesion or "intact", **{k: float(v.mean()) for k, v in metrics.items()}})
    # Save deterministic traces for the representative median-validation training seed.
    chosen_run = sorted([r for r in selection["runs"] if r["family"] == primary], key=lambda r: r["validation_cost"])[1]
    trace_seed = chosen_run["seed"]
    for name in (primary, "feedback", "balance", "off"):
        policy = HouseBrain.load(CHECKPOINTS / f"{name}-s{trace_seed}.pt") if name == primary else name
        metrics, trace = rollout(policy, scenarios([TEST[0]], 288), trace=True)
        write_json(REPORTS / f"trace-{name}.json", {"controller": name, "seed": trace_seed if name == primary else 0,
                   "scenario_seed": TEST[0], "metrics": {k: float(v[0]) for k, v in metrics.items()}, "trace": trace})
    result = {"selection_sha256": selection_sha, "primary": primary, "representative_seed": trace_seed,
              "summaries": summaries, "comparisons": comparisons, "timings": timings,
              "interventions": interventions, "evaluation_seconds": time.perf_counter() - start,
              "episodes": len(rows), "transitions": len(rows) * 288,
              "test_opened_once": True, "selection_changed_after_test": False}
    write_json(REPORTS / "results.json", result)
    print(f"Saved {len(rows)} episodes; primary={primary}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate", "smoke"))
    args = parser.parse_args()
    if args.command == "train": train_study()
    elif args.command == "evaluate": evaluate_study()
    else:
        torch.set_num_threads(2)
        b = HouseBrain()
        opt = torch.optim.Adam(b.parameters(), lr=.003)
        before = b.recurrent.detach().clone()
        start = time.perf_counter()
        for _ in range(3):
            e = scenarios(list(range(1200, 1224)), 24, training=True)
            s = e.initial; loss = torch.zeros(())
            opt.zero_grad()
            for t in range(24):
                s, m = transition(s, b(observe(s, e, t)), e, t)
                loss = loss + m["cost"].mean() / 24
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(b.parameters(), 5, error_if_nonfinite=True)
            opt.step()
        print(json.dumps({"seconds_per_update": (time.perf_counter() - start) / 3,
                          "loss": float(loss.detach()), "gradient_norm": float(norm),
                          "recurrent_changed": not torch.equal(before, b.recurrent), "stats": b.stats()}, indent=2))


if __name__ == "__main__": main()
