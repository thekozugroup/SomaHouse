"""Study 2: standalone imitation, then reward optimization on fresh splits."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time

import numpy as np
import torch

from .brain import ROOT, HouseBrain
from .sim import actuate, baseline, observe, rollout, scenarios, transition
from .study import rows_for, sha, write_json

OUT = ROOT / "reports/teacher"
CKPT = ROOT / "checkpoints/teacher"
PROTOCOL = ROOT / "research/teacher-protocol.md"
SEEDS = [107, 119, 141]
VAL = list(range(6_000_001, 6_000_033))
TEST = list(range(7_000_001, 7_000_101))
STRESS = list(range(8_000_001, 8_000_025))
PRIMARY = "fly-teacher-reward"


class Teacher:
    def __init__(self, config):
        self.config = config

    def __call__(self, obs, **kwargs):
        if self.config.get("original"):
            return baseline(obs, "balance")
        temp, co2, people = obs[:, :3] * 8 + 22, obs[:, 3:6] * 1000 + 420, obs[:, 9:12] * 3
        target, day, outside = obs[:, 12:15] * 4 + 22, obs[:, 15:18] * 600, obs[:, 18:19] * 20 + 15
        occ = (people > 0).float()
        desired = target + self.config["offset"] * torch.sign(outside - target) * occ
        vent = ((18000 / 82 * people / (self.config["co2_target"] - 420) - .12) / 3.4
                + self.config["vent_gain"] * (co2 - self.config["co2_target"]) / 1200).clamp(0, 1)
        feed = -(.075 + .000335 * 82 * (.12 + 3.4 * vent)) * (outside - temp)
        feed -= .12 * people + .00035 * day
        heat = (feed + self.config["gain"] * (desired - temp) * (.12 + .88 * occ)) / 3
        light = ((300 - day) / 450).clamp(0, 1) * occ
        return torch.cat((heat.clamp(-1, 1), vent, light), 1)


@torch.no_grad()
def validation(policy):
    metrics, _ = rollout(policy, scenarios(VAL, 288))
    return {k: float(v.mean()) for k, v in metrics.items()}


def tune_teacher():
    path = OUT / "teacher.json"
    if path.exists():
        return json.loads(path.read_text())
    configs = [{"original": True}] + [dict(zip(("gain", "offset", "co2_target", "vent_gain"), v))
        for v in itertools.product((.65, 1.3, 2.), (0., .4, .75), (850., 950.), (.6, 1.2))]
    # The grid has 36 members: 3*3*2*2, plus the unmodified study-1 reference.
    measured = [{"config": c, "validation": validation(Teacher(c))} for c in configs]
    best = min(measured, key=lambda r: r["validation"]["cost"])
    data = {"selected": best, "candidates": measured, "protocol_sha256": sha(PROTOCOL),
            "validation_seeds": VAL, "selected_without_test": True}
    write_json(path, data)
    print(f"Teacher selected: {best}", flush=True)
    return data


@torch.no_grad()
def collect(policy, teacher, episode_seeds, steps):
    env = scenarios(episode_seeds, steps, training=True)
    state = env.initial
    observations, targets = [], []
    for t in range(steps):
        obs = observe(state, env, t)
        target = actuate(teacher(obs))
        observations.append(obs.detach())
        targets.append(target.detach())
        state, _ = transition(state, policy(obs), env, t)
    return torch.cat(observations), torch.cat(targets)


def imitation(model, teacher, seed, family):
    ident = f"{family}-imitation-s{seed}"
    path, meta = CKPT / f"{ident}.pt", OUT / f"{ident}.json"
    if meta.exists():
        data = json.loads(meta.read_text()); assert sha(path) == data["sha256"]
        return data
    initial = CKPT / f"{family}-initial-s{seed}.pt"
    model.save(initial)
    rng = np.random.default_rng(seed + 1700)
    obs, targets = collect(teacher, teacher, rng.integers(5_000_000, 6_000_000, 1024).tolist(), 24)
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    v = validation(model); best, selected = v["cost"], 0
    model.save(path)
    curve = [{"update": 0, "validation": v}]
    start = time.perf_counter()
    for update in range(1, 401):
        idx = torch.from_numpy(rng.integers(0, len(obs), 512))
        pred = actuate(model(obs[idx]))
        loss = (((pred - targets[idx]) ** 2) * torch.tensor([1.] * 6 + [4.] * 3)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if update == 200:
            extra = collect(model, teacher, rng.integers(5_000_000, 6_000_000, 256).tolist(), 12)
            obs, targets = torch.cat((obs, extra[0])), torch.cat((targets, extra[1]))
        if update % 100 == 0:
            v = validation(model)
            curve.append({"update": update, "imitation_loss": float(loss.detach()), "validation": v})
            if v["cost"] < best:
                best, selected = v["cost"], update; model.save(path)
            print(f"{ident} {update}/400 validation={v['cost']:.5f} best={best:.5f}", flush=True)
    data = {"id": ident, "family": f"{family}-imitation", "seed": seed,
            "stage": "imitation", "updates": 400, "selected_update": selected,
            "validation_cost": best, "curve": curve, "training_seconds": time.perf_counter() - start,
            "supervised_pairs_presented": 400 * 512, "labelled_states": len(obs),
            "dataset_transitions": 1024 * 24 + 256 * 12,
            "checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path),
            "initial_checkpoint": str(initial.relative_to(ROOT)), "initial_sha256": sha(initial),
            "stats": HouseBrain.load(path).stats()}
    write_json(meta, data)
    return data


def reward_loss(metrics, state, env, t):
    occ = (env.people[t] > 0).float()
    air = (.5 * (torch.relu(state.co2 - 950) / 500) ** 2 * occ).mean()
    light = (2 * (torch.relu(300 - metrics["lux"]) / 300) ** 2 * occ).mean()
    return metrics["cost"].mean() + air + light


def reward(model, seed, family, updates, parent):
    ident = f"{family}-s{seed}"
    path, meta = CKPT / f"{ident}.pt", OUT / f"{ident}.json"
    if meta.exists():
        data = json.loads(meta.read_text()); assert sha(path) == data["sha256"]
        return data
    rng = np.random.default_rng(seed + 2900)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    v = validation(model); best, selected = v["cost"], 0
    model.save(path); curve = [{"update": 0, "validation": v}]
    start = time.perf_counter()
    for update in range(1, updates + 1):
        env = scenarios(rng.integers(5_000_000, 6_000_000, 32).tolist(), 48, training=True)
        state = env.initial; loss = torch.zeros(())
        opt.zero_grad(set_to_none=True)
        for t in range(48):
            state, metrics = transition(state, model(observe(state, env, t)), env, t)
            loss = loss + reward_loss(metrics, state, env, t) / 48
        if not torch.isfinite(loss): raise RuntimeError(f"Non-finite training {ident}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if update % 100 == 0 or update == updates:
            v = validation(model)
            curve.append({"update": update, "training_objective": float(loss.detach()), "validation": v})
            if v["cost"] < best:
                best, selected = v["cost"], update; model.save(path)
            print(f"{ident} {update}/{updates} validation={v['cost']:.5f} best={best:.5f}", flush=True)
    data = {"id": ident, "family": family, "seed": seed, "stage": "reward",
            "updates": updates, "selected_update": selected, "validation_cost": best,
            "curve": curve, "training_seconds": time.perf_counter() - start,
            "training_transitions": updates * 32 * 48, "parent_checkpoint": parent,
            "checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path),
            "stats": HouseBrain.load(path).stats()}
    write_json(meta, data); return data


def train():
    torch.set_num_threads(2); OUT.mkdir(exist_ok=True); CKPT.mkdir(exist_ok=True)
    if (OUT / "results.json").exists(): raise RuntimeError("Final study 2 already opened")
    teacher_data = tune_teacher(); teacher = Teacher(teacher_data["selected"]["config"])
    runs = []
    for seed in SEEDS:
        for family, topology in (("fly", "fly"), ("dense", "dense")):
            initial = HouseBrain(32, seed, topology=topology, copies=2)
            imitation_run = imitation(initial, teacher, seed, family); runs.append(imitation_run)
            model = HouseBrain.load(ROOT / imitation_run["checkpoint"])
            runs.append(reward(model, seed, f"{family}-teacher-reward", 800, imitation_run["checkpoint"]))
            if family == "fly":
                model = HouseBrain.load(ROOT / imitation_run["initial_checkpoint"])
                runs.append(reward(model, seed, "fly-reward-only", 1200, imitation_run["initial_checkpoint"]))
    representative = sorted([r for r in runs if r["family"] == PRIMARY], key=lambda r: r["validation_cost"])[1]
    write_json(OUT / "selection.json", {"primary": PRIMARY, "representative_seed": representative["seed"],
         "teacher": teacher_data["selected"], "teacher_sha256": sha(OUT / "teacher.json"),
         "protocol_sha256": sha(PROTOCOL), "training_seeds": SEEDS,
         "validation_seeds": VAL, "test_seeds": TEST, "stress_seeds": STRESS,
         "runs": runs, "selected_before_test": True})
    print(f"SEALED STUDY 2 primary={PRIMARY} representative={representative['seed']}", flush=True)


def paired(rows, a, b, key):
    def grouped(name):
        groups = {}
        for r in rows:
            if r["controller"] == name and r["stress"] == "normal":
                groups.setdefault(r["scenario_seed"], []).append(r[key])
        return {s: float(np.mean(v)) for s, v in groups.items()}
    left, right = grouped(a), grouped(b)
    keys = sorted(left.keys() & right.keys()); assert len(keys) == len(TEST)
    delta = np.array([left[k] - right[k] for k in keys])
    rng = np.random.default_rng(1907)
    boot = delta[rng.integers(0, len(keys), (5000, len(keys)))].mean(1)
    return {"a": a, "b": b, "metric": key, "difference": float(delta.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "paired_scenarios": len(keys)}


@torch.no_grad()
def evaluate():
    torch.set_num_threads(2)
    if (OUT / "results.json").exists(): raise RuntimeError("Evaluation exists; preserve immutable evidence")
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha(PROTOCOL)
    assert selection["teacher_sha256"] == sha(OUT / "teacher.json")
    policies = [("teacher", 0, Teacher(selection["teacher"]["config"])), ("balance", 0, "balance")]
    for run in selection["runs"]:
        assert sha(ROOT / run["checkpoint"]) == run["sha256"]
        policies.append((run["family"], run["seed"], HouseBrain.load(ROOT / run["checkpoint"])))
    rows = []; start = time.perf_counter()
    for name, seed, policy in policies:
        for stress, seeds in [("normal", TEST)] + [(s, STRESS) for s in ("weather", "temperature_bias", "co2_dropout", "capacity")]:
            env = scenarios(seeds, 288, stress=stress)
            m, _ = rollout(policy, env)
            rows.extend(rows_for(name, seed, m, env, "test" if stress == "normal" else "stress"))
        print(f"Study 2 evaluated {name} s{seed}", flush=True)
    summaries = []
    for name in sorted({r["controller"] for r in rows}):
        for stress in ("normal", "weather", "temperature_bias", "co2_dropout", "capacity"):
            subset = [r for r in rows if r["controller"] == name and r["stress"] == stress]
            summaries.append({"controller": name, "stress": stress, "episodes": len(subset),
                 **{k: float(np.mean([r[k] for r in subset])) for k, v in subset[0].items() if isinstance(v, float)}})
    keys = ("cost", "energy_kwh", "thermal_compliance", "co2_compliance", "light_compliance")
    comparisons = [paired(rows, PRIMARY, "teacher", k) for k in keys]
    comparisons += [paired(rows, PRIMARY, b, "cost") for b in ("balance", "fly-imitation", "fly-reward-only", "dense-teacher-reward")]
    normal = {r["controller"]: r for r in summaries if r["stress"] == "normal"}
    cost = comparisons[0]; energy = comparisons[1]
    per_seed = [{"seed": s, "cost": float(np.mean([r["cost"] for r in rows if r["controller"] == PRIMARY and r["training_seed"] == s and r["stress"] == "normal"]))} for s in SEEDS]
    gates = {"cost_at_least_2pct_lower": normal[PRIMARY]["cost"] <= normal["teacher"]["cost"] * .98,
             "cost_ci_below_zero": cost["ci95"][1] < 0,
             "all_training_seeds_lower": all(r["cost"] < normal["teacher"]["cost"] for r in per_seed),
             "energy_ci_within_2pct": energy["ci95"][1] < normal["teacher"]["energy_kwh"] * .02}
    gates.update({f"{k}_ci_within_1pp": next(c for c in comparisons if c["metric"] == k)["ci95"][0] > -.01 for k in keys[2:]})
    with (OUT / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    for name in (PRIMARY, "teacher"):
        policy = Teacher(selection["teacher"]["config"]) if name == "teacher" else HouseBrain.load(CKPT / f"{PRIMARY}-s{selection['representative_seed']}.pt")
        m, trace = rollout(policy, scenarios([TEST[0]], 288), trace=True)
        write_json(OUT / f"trace-{name}.json", {"controller": name, "seed": selection["representative_seed"] if name == PRIMARY else 0,
                   "scenario_seed": TEST[0], "metrics": {k: float(v[0]) for k, v in m.items()}, "trace": trace})
    write_json(OUT / "results.json", {"primary": PRIMARY, "representative_seed": selection["representative_seed"],
            "selection_sha256": sha(OUT / "selection.json"), "selection_changed_after_test": False,
            "summaries": summaries, "comparisons": comparisons, "primary_per_seed": per_seed,
            "gates": gates, "outperforms_teacher": all(gates.values()),
            "episodes": len(rows), "transitions": len(rows) * 288,
            "evaluation_seconds": time.perf_counter() - start})
    print(json.dumps({"gates": gates, "primary": normal[PRIMARY], "teacher": normal["teacher"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("train", "evaluate"))
    args = parser.parse_args()
    (train if args.command == "train" else evaluate)()
