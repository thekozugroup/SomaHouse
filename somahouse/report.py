"""Build publication figures, detailed report and a portable static case study."""
from __future__ import annotations

import csv
import html
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .brain import ROOT
from .study import REPORTS, sha, write_json

LABELS = {"duplicate": "Duplicated circuit", "continue": "Base + extra training", "cut": "Pruned circuit",
          "fly32": "Base fly32", "fly64": "Larger fly64", "rewired32": "Rewired32", "dense32": "Dense32",
          "nohub32": "No hub", "feedback": "Feedback control", "balance": "Balance control",
          "off": "Equipment off", "untrained32": "Untrained circuit"}
COLORS = {"duplicate": "#aa5b37", "continue": "#8d9978", "cut": "#bf9a67", "fly32": "#789068",
          "fly64": "#6c8e99", "rewired32": "#9c929b", "dense32": "#66716a", "nohub32": "#b6a98b",
          "feedback": "#a59977", "balance": "#385745"}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c0c6b7", "text.color": "#2b4033",
                         "axes.labelcolor": "#52664f", "xtick.color": "#61705a", "ytick.color": "#61705a",
                         "figure.facecolor": "#f5f3eb", "axes.facecolor": "#f5f3eb", "grid.color": "#dbe0d2",
                         "savefig.facecolor": "#f5f3eb", "figure.dpi": 130})


def save(fig, name):
    fig.savefig(REPORTS / name, bbox_inches="tight", dpi=160)
    plt.close(fig)


def figures(selection, result, rows):
    setup()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for name in ("fly32", "fly64", "rewired32", "dense32", "nohub32"):
        runs = [r for r in selection["runs"] if r["family"] == name]
        updates = [x["update"] for x in runs[0]["curve"]]
        values = np.array([[c["validation"]["cost"] for c in r["curve"]] for r in runs])
        ax.plot(updates, values.mean(0), label=LABELS[name], color=COLORS[name], lw=2)
        ax.fill_between(updates, values.min(0), values.max(0), color=COLORS[name], alpha=.1)
    ax.set_yscale("log"); ax.grid(axis="y", alpha=.65)
    ax.set(xlabel="Optimizer updates", ylabel="Validation cost · lower is better",
           title="Learning from consequences inside the house")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "learning.png")

    normal = {r["controller"]: r for r in result["summaries"] if r["stress"] == "normal"}
    order = ["balance", "feedback", "duplicate", "continue", "cut", "fly64", "fly32", "dense32", "rewired32", "nohub32"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, key, label in zip(axes, ["cost", "energy_kwh", "thermal_degree_hours"],
                             ["Weighted cost", "Electricity · kWh/day", "Occupied discomfort · °C·h/zone/day"]):
        v = [normal[n][key] / (2 if key != "cost" else 1) for n in order]
        ax.barh(np.arange(len(order)), v, color=[COLORS[n] for n in order], height=.62)
        ax.set_yticks(np.arange(len(order)), [LABELS[n] for n in order], fontsize=9)
        ax.set_xlabel(label, fontsize=9); ax.grid(axis="x", alpha=.4); ax.set_axisbelow(True)
    axes[0].invert_yaxis(); fig.suptitle("A useful learned policy, with a stronger conventional reference", x=.52)
    fig.tight_layout(); save(fig, "comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for name in ("cut", "fly32", "continue", "fly64", "duplicate"):
        runs = [r for r in selection["runs"] if r["family"] == name]
        x = runs[0]["stats"]["spoke_units"]
        val = [r["validation_cost"] for r in runs]
        test = [np.mean([float(row["cost"]) for row in rows if row["controller"] == name and row["stress"] == "normal" and int(row["training_seed"]) == r["seed"]]) for r in runs]
        for ax, values in zip(axes, (val, test)):
            offset = {"continue": -4, "fly32": 4, "fly64": -5, "duplicate": 5}.get(name, 0)
            ax.scatter([x + offset] * 3, values, color=COLORS[name], s=23, alpha=.7)
            ax.scatter([x + offset], [np.mean(values)], color=COLORS[name], marker="_", s=250)
            ax.plot([], [], color=COLORS[name], marker="o", ls="", label=LABELS[name])
    for ax, title in zip(axes, ("Validation · used for selection", "Held-out test · opened after selection")):
        ax.set(xlabel="Spoke units · hub adds 16", ylabel="Cost", title=title, xticks=[64, 128, 256]); ax.grid(alpha=.4)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "capacity.png")

    shifts = ["normal", "weather", "temperature_bias", "co2_dropout", "capacity"]
    names = ["balance", "feedback", "duplicate", "continue", "cut"]
    matrix = np.array([[next(r["cost"] for r in result["summaries"] if r["controller"] == n and r["stress"] == s) for s in shifts] for n in names])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.imshow(np.log10(matrix + .05), cmap="YlOrBr", aspect="auto")
    for i in range(len(names)):
        for j in range(len(shifts)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="#27352d" if matrix[i,j] < 1.7 else "white")
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names]); ax.set_xticks(range(5), ["Normal", "Severe weather", "+2°C sensor bias", "CO₂ stuck at 650", "45% capacity"], fontsize=9)
    ax.set_title("Failures expose what the reward alone can hide · cost, lower is better", pad=15)
    fig.tight_layout(); save(fig, "robustness.png")

    primary = json.loads((REPORTS / f"trace-{result['primary']}.json").read_text())
    balance = json.loads((REPORTS / "trace-balance.json").read_text())
    t = [r["hour"] for r in primary["trace"]]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for zone, color in enumerate(("#ad6240", "#718a62", "#648b95")):
        axes[0].plot(t, [r["temperature"][zone] for r in primary["trace"]], color=color, label=("Living", "Bedroom", "Office")[zone], lw=1.3)
        axes[0].plot(t, [r["setpoint"][zone] for r in primary["trace"]], color=color, ls="--", lw=.7, alpha=.6)
        axes[1].plot(t, [r["co2"][zone] for r in primary["trace"]], color=color, lw=1.2)
        axes[2].plot(t, [r["lux"][zone] for r in primary["trace"]], color=color, lw=1.2)
    axes[1].axhline(1000, ls="--", color="#9c8156", lw=.8); axes[2].axhline(300, ls="--", color="#9c8156", lw=.8)
    axes[3].plot(t, [r["power_kw"] for r in primary["trace"]], color=COLORS["duplicate"], label="Learned")
    axes[3].plot(t, [r["power_kw"] for r in balance["trace"]], color=COLORS["balance"], label="Balance", alpha=.8)
    for ax, label in zip(axes, ("Temperature · °C", "CO₂ · ppm", "Illumination · lux", "Power · kW")):
        ax.set_ylabel(label); ax.grid(alpha=.4)
    axes[0].legend(frameon=False, ncol=3, fontsize=8); axes[3].legend(frameon=False, fontsize=8)
    axes[3].set_xlabel("Simulated hour"); axes[3].set_xlim(0,48)
    fig.suptitle(f"Representative trace · seed {result['representative_seed']} · scenario 3000001\nSelected by validation rank; scenario fixed before testing", fontsize=11)
    fig.tight_layout(); save(fig, "response.png")

    lesions = ["hub", "context", "air", "light", "recurrence"]
    changes = []
    for l in lesions:
        deltas = []
        for seed in selection["training_seeds"]:
            intact = next(r["cost"] for r in result["interventions"] if r["seed"] == seed and r["lesion"] == "intact")
            cut = next(r["cost"] for r in result["interventions"] if r["seed"] == seed and r["lesion"] == l)
            deltas.append((cut / intact - 1) * 100)
        changes.append(deltas)
    fig, ax = plt.subplots(figsize=(9,3.6))
    ax.bar(range(len(lesions)), np.mean(changes, 1), color="#8b9e78", width=.55)
    for i, values in enumerate(changes): ax.scatter([i] * 3, values, color="#ac5b36", s=18, zorder=4)
    ax.set_xticks(range(len(lesions)), ["Hub removed", "Context removed", "Air removed", "Light removed", "Recurrence removed"])
    ax.axhline(0,color="#849177",lw=.8); ax.set_ylabel("Validation cost increase · %"); ax.set_title("The hub is used; not every spoke is equally important")
    ax.grid(axis="y", alpha=.35); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "interventions.png")


def report():
    selection = json.loads((REPORTS / "selection.json").read_text())
    result = json.loads((REPORTS / "results.json").read_text())
    verification = json.loads((REPORTS / "verification.json").read_text())
    rows = list(csv.DictReader((REPORTS / "episodes.csv").open()))
    normal = {r["controller"]: r for r in result["summaries"] if r["stress"] == "normal"}
    primary = result["primary"]; learned = normal[primary]; base = normal["continue"]; balance = normal["balance"]
    improvement = 100 * (1 - learned["cost"] / base["cost"])
    energy_extra = 100 * (learned["energy_kwh"] / balance["energy_kwh"] - 1)
    latency = np.median([r["median_ms"] for r in result["timings"] if r["controller"] == primary])
    training_seconds = sum(r["training_seconds"] for r in selection["runs"])
    stats = next(r["stats"] for r in selection["runs"] if r["family"] == primary)
    figures(selection, result, rows)
    table = "| Controller | Cost ↓ | kWh/day ↓ | °C·h/zone/day ↓ | CO₂ ppm·h/zone/day ↓ | Light lux·h/zone/day ↓ | Temp / CO₂ / light compliance |\n|---|---:|---:|---:|---:|---:|---|\n"
    for name in ("balance", "feedback", "duplicate", "continue", "cut", "fly64", "fly32", "dense32", "rewired32", "nohub32", "untrained32", "off"):
        r = normal[name]
        table += f"| {LABELS[name]} | {r['cost']:.4f} | {r['energy_kwh']/2:.2f} | {r['thermal_degree_hours']/2:.3f} | {r['co2_ppm_hours']/2:.2f} | {r['light_lux_hours']/2:.2f} | {r['thermal_compliance']*100:.1f}% / {r['co2_compliance']*100:.1f}% / {r['light_compliance']*100:.1f}% |\n"
    ci_table = "| A minus B, cost | Mean difference | Paired scenario 95% interval |\n|---|---:|---|\n"
    for c in result["comparisons"]:
        ci_table += f"| {LABELS[c['a']]} − {LABELS[c['b']]} | {c['difference']:+.5f} | [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] |\n"
    stress_table = "| Shift | Controller | Cost | kWh/day | Temperature compliance | CO₂ compliance |\n|---|---|---:|---:|---:|---:|\n"
    for stress in ("weather", "temperature_bias", "co2_dropout", "capacity"):
        for name in ("balance", primary):
            r = next(r for r in result["summaries"] if r["controller"] == name and r["stress"] == stress)
            stress_table += f"| {stress} | {LABELS[name]} | {r['cost']:.3f} | {r['energy_kwh']/2:.2f} | {r['thermal_compliance']*100:.1f}% | {r['co2_compliance']*100:.1f}% |\n"
    seed_table = "| Family | Seed | Selected update | Validation cost | Test cost |\n|---|---:|---:|---:|---:|\n"
    seed_rows = []
    for r in selection["runs"]:
        avg = np.mean([float(row["cost"]) for row in rows if row["controller"] == r["family"] and row["stress"] == "normal" and int(row["training_seed"]) == r["seed"]])
        seed_table += f"| {r['family']} | {r['seed']} | {r['selected_update']} | {r['validation_cost']:.5f} | {avg:.5f} |\n"
        seed_rows.append({"family":r["family"],"seed":r["seed"],"selected_update":r["selected_update"],"validation_cost":r["validation_cost"],"test_cost":float(avg)})
    with (REPORTS / "seed-results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(seed_rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(seed_rows)
    anatomy = "| Source class lesion | Intact mean validation cost | Lesioned mean cost | Interpretation |\n|---|---:|---:|---|\n"
    intact = np.mean([r["cost"] for r in result["interventions"] if r["lesion"] == "intact"])
    for lesion in sorted({r["lesion"] for r in result["interventions"] if r["lesion"].startswith("class:")}):
        v = np.mean([r["cost"] for r in result["interventions"] if r["lesion"] == lesion])
        anatomy += f"| {lesion[6:]} | {intact:.4f} | {v:.4f} | Computational lesion; no claim about fly behavior |\n"
    text = f"""# SomaHouse: teaching a fly-derived circuit to regulate a home

Michael Wong · 22 October 2026 · Study 1

## Findings in plain terms

The controller learned to regulate a simulated three-room home. Doubling its functional spokes improved held-out cost by **{improvement:.1f}%** compared with giving the smaller circuit the same extra training. Cutting each spoke to half its cells made control worse. That makes duplication a useful result in this specific experiment, rather than a reason to assume that bigger biological circuits are always better.

The stronger conventional balance controller remained the better practical reference: lower average energy, less temperature discomfort and better CO₂ and light compliance. The learned controller used **{energy_extra:.1f}% more electricity** than that reference. Their aggregate cost difference was uncertain, and that aggregate conceals different comfort tradeoffs. Dense artificial wiring also outperformed the same-size fly-derived base. This study does not establish a special advantage from biological connectivity.

## Design and provenance

SomaHouse is isolated from FlyFi image compression: a separate repository, package, environment, server port, simulator, training runs and artifacts. Only a public, attributed connectome seed was copied. Image data and trained image models are not used.

The source is MaleCNS v1.0, attributed to FlyEM at HHMI Janelia, University of Cambridge, MRC LMB and Google Research, CC BY 4.0. The included 512-cell seed was previously extracted by connected greedy growth from a 5,609-cell learning-circuit annotation pool. Original source URLs and SHA-256 fingerprints, exact source IDs and extraction method are in data/connectome/provenance.json. This experiment uses nested 32- and 64-cell subsets, or salience-selected 16-cell descendants. These are not the complete fly brain, an unbiased anatomical sample, or all regions needed for biological behavior.

Each source subset initializes four independent artificial spokes: thermal, air, light and context. Sensor masks assign different observation streams. Every spoke sends an 8-number message to a learned 16-unit hub. The hub receives four concatenated domain ports and broadcasts back through private receive matrices, twice per decision. With duplicated spokes, messages are averaged within each domain, preserving four hub ports; split readouts sum the copies. Direct spoke-to-spoke links do not exist. The context spoke has no direct actuator head.

The selected model has **{stats['modules']} spokes × 32 cells = {stats['spoke_units']} spoke units**, plus 16 artificial hub units, **{stats['source_unique_ids']} unique source cell IDs**, {stats['recurrent_edges']:,} masked recurrent edges and {stats['allocated_parameters']:,} allocated scalar parameters. All functional spokes already reuse the same source subset; doubling those spokes adds artificial copies, not newly observed source cells. Source GABA/glutamate labels give an initialization sign heuristic only; optimization can change weight signs. Tanh dynamics, adapters, hub and readouts are engineered. No electrophysiology is simulated. The term cortex is intentionally avoided: this hub is an engineering component.

The network is reactive between decisions. Its recurrent computation is internal to a decision, with two updates; the next decision observes current sensors and the previous executed action. It has no cross-step learned hidden state, no language model and no autonomous biological development.

## Physical model and assumptions

Three zones represent living, bedroom and office. The control interval is Δt = 1/6 hour. For zone i, temperature T, CO₂ concentration C, outdoor temperature To and action u:

```
ACH_i = 0.12 + 3.4 × vent_i
Q_i = G_i(To − T_i) + 0.000335 V_i ACH_i(To − T_i)
      + 0.025(mean(T) − T_i) + 0.12 people_i
      + 0.00035 daylight_i + 0.07 light_i + 3 heat_i
T_i(next) = T_i + Δt Q_i / heat_capacity_i
C_i(next) = C_i + Δt (18000 people_i / V_i − ACH_i(C_i − 420))
lux_i = daylight_i + 450 light_i
power = sum_i(3 abs(heat_i)/3.2 + 0.045 vent_i³ + 0.07 light_i)
```

Heat is signed for heating or cooling, bounded to ±3 kW per zone and a shared 6 kW thermal limit. The fixed assumed heat-pump COP is 3.2. Vent and light fractions are 0–1. The equipment-off control has passive infiltration and no electrical use; standby and fan energy for that passive infiltration are not modeled. Reduced-capacity stress scales HVAC and vent capacity to 45%, leaving lights unchanged. Interzone heat exchange is symmetric, while interzone CO₂ transfer is not modeled.

Per-scenario heat capacity varies 1.3–2.2 kWh/K, envelope conductance 0.055–0.095 kW/K and room volume 65–100 m³. Three equally sampled synthetic seasons have average outdoor temperatures near 3, 17 and 31°C with daily waves and correlated disturbances. Daylight follows a day/night curve with time-varying cloud modulation. Occupancy schedules vary in timing and include visits. Temperature targets vary by room and shift after a randomized daytime point. Initial temperatures are within ±2.5°C of target; initial CO₂ is 450–1,100 ppm. These are plausible toy-system assumptions, not a calibrated house or real weather dataset.

Current observations contain temperature, CO₂, illumination, people, targets, daylight, outdoor temperature, time-of-day, previous actions and a synthetic time-of-use price multiplier. No controller sees future weather or occupancy. Neural controllers do not receive randomized true plant parameters. Conventional balance control uses fixed nominal parameters (82 m³, 0.075 kW/K and a nominal tracking gain); its model-based engineering knowledge is an explicit advantage. This is not a compute- or information-prior-matched comparison against an optimized MPC system.

## Training, reward and selection

Training uses analytical gradients through the differentiable simulator to minimize multi-step control cost. It starts with random adapters and source-initialized recurrent weights; no imitation targets are supplied. This is model-based policy learning, not model-free PPO or evidence of autonomous online learning. Knowing simulator derivatives substantially simplifies learning.

Per-step cost is the mean across three rooms of 4 × occupied temperature slack², 0.4 × empty-room temperature slack², 0.05 × occupied tracking error², 2 × occupied (CO₂ excess/500)² and 4 × occupied (light deficit/300)², plus 0.08 × electrical kW × price and 0.012 × mean absolute action movement. Occupied thermal slack starts outside ±1°C; empty-room slack outside ±3°C. CO₂ excess starts at 1,000 ppm and light deficit below 300 lux. These are engineering targets, not universal health or comfort standards. Soft penalties permit violations; no hard safety guarantee is imposed. This matters for interpreting the CO₂ result.

Three training seeds (7, 19, 41) each trained five base architectures for 300 updates: fly32, fly64, degree-preserving rewired32, dense32 and nohub32. All got the same exogenous training draws per seed and optimizer schedule. Each update used 24 independently generated four-hour windows, starting at randomized times and states; truncated windows are not complete day episodes. Adam learning rate was 0.003, gradients clipped at norm 5.

Each trained fly32 produced continue, cut and duplicate descendants, each receiving 150 additional updates at learning rate 0.0015 and matched draws. Pruning uses combined input/readout/send/receive weight salience averaged across modules to retain 16 source IDs; actual tensors are compacted. This heuristic is not a search for optimal anatomical regions. Duplication initially preserves actions through mean hub pooling and split readouts; small input perturbations then break copy symmetry. Continue supplies the equal-extra-training control. Comparing duplication only to the 300-update base would confound more training and more capacity.

The total is **24 trained candidates, 5,850 optimizer updates, 140,400 windows and 3,369,600 training transitions**, plus validation rollouts. Summed run time was {training_seconds/60:.1f} minutes on an M1 Max with 32 GiB, CPU, two PyTorch threads. Validation used 24 fixed independent 24-hour scenarios, at update 0, every 50 updates and the final update. Each candidate retains its lowest-cost validation checkpoint, including step zero if needed. Initial and selected checkpoints are saved.

Training, validation, primary test and stress seeds occupy disjoint ranges. The committed protocol predates training; selection was committed at Git revision 4e527ee before test evaluation. The primary family was selected from fly32, fly64, continue, cut and duplicate by mean validation cost, preferring fewer units within 2% of the best. Duplication additionally had to improve all three validation seeds. It met those conditions: validation mean {selection['family_validation_costs']['duplicate']:.5f} versus {selection['family_validation_costs']['continue']:.5f} for continue. A held-out paired interval also had to exclude zero before a duplication benefit could be claimed. No test-based tuning followed.

![Validation learning curves](learning.png)

## Held-out results

The primary set contains 60 independent 48-hour scenarios. Learned families each contribute 180 episodes across three training seeds. Conventional policies each contribute 60 episodes. Four stress families add 16 independent scenario seeds each; stress seeds are reused across shifts for pairing. Across 30 policy instances, the evaluation contains **3,720 episodes and 1,071,360 transitions**. These cover 76 unique exogenous seed IDs and 124 scenario/shift combinations; they are not 3,720 independent buildings.

Daily metrics divide 48-hour totals by two. Temperature, CO₂ and light deficits are integrated over occupied zones then divided by all three zones, not only occupied zones; compliance fractions use occupied zone-hours. Means of scenario compliance fractions are reported. Light compliance permits a 5-lux numerical/practical margin (≥295 lux), while deficit remains measured below 300 lux. That margin was present in the simulator before training. The weighted objective includes unoccupied drift, price and movement, so it cannot be reconstructed from the daily occupied deficits alone.

{table}
![Control comparison](comparison.png)

The selected duplicated controller reaches **{learned['thermal_compliance']*100:.1f}% temperature, {learned['co2_compliance']*100:.1f}% CO₂ and {learned['light_compliance']*100:.1f}% light compliance**, compared with {balance['thermal_compliance']*100:.1f}%, {balance['co2_compliance']*100:.2f}% and {balance['light_compliance']*100:.1f}% for balance control. Daily energy is {learned['energy_kwh']/2:.2f} versus {balance['energy_kwh']/2:.2f} kWh. It beats the proportional feedback baseline on weighted cost and thermal regulation, while spending more electricity and allowing more CO₂ excess. These are tradeoffs, not blanket energy savings.

## Capacity, topology and uncertainty

The matched duplication gain is {improvement:.2f}%. Its paired scenario cost interval excludes zero. Pruning increases mean cost by {100*(normal['cut']['cost']/base['cost']-1):.1f}% versus continue, despite fewer parameters. The larger 64-cell source subset and the duplicated 32-cell subset both have 256 spoke units but differ in unique cells and training schedule; they do not isolate anatomy alone.

{ci_table}

Intervals use 2,000 paired bootstrap resamples over the 60 test scenarios after averaging training seeds, with bootstrap seed 918. Negative means A is lower-cost. The training seeds are treated as three fixed replicates, so these intervals understate uncertainty about arbitrary retraining. All per-seed outcomes are below. Controls are predeclared; the secondary comparison intervals are not adjusted for multiple testing. A significant fly-versus-rewired difference is small and does not prove a general connectome advantage, especially because dense wiring does better and has more active edges. Dense and fly controls have the same allocated tensor dimensions; the random control preserves exact directed in/out degrees through double-edge swaps, not merely a node permutation.

![Capacity tradeoffs](capacity.png)

{seed_table}

## What communication and lesions reveal

Zeroing hub output, a functional spoke, recurrence, or a source cell class is a post-hoc validation intervention; no lesion is retrained. Hub removal raises cost in all three selected models. The context spoke has a small and inconsistent effect, including one slight improvement when removed; this module is a candidate for future retraining/pruning, not a proven necessary component. Thermal lesions are catastrophic because the thermal head loses its learned state. Air, light and recurrence lesions worsen performance. Source class lesions mainly identify where this engineered task's computation ended up, not what those cells do in a fly.

![Intervention sensitivity](interventions.png)

{anatomy}

## Robustness and failure modes

{stress_table}

Sensor bias is a major failure: temperature compliance collapses for both controllers. Reduced equipment capacity exposes physical limits; learned control has a lower aggregate penalty here, but still violates targets substantially. Under a CO₂ sensor stuck at 650 ppm, the learned objective can improve despite worse CO₂ compliance because the soft aggregate trades ventilation and thermal/energy costs. This is a concrete example of why a reward reduction is insufficient evidence of acceptable home control. No fault detector, redundant sensing or independently enforced safety supervisor exists in this prototype.

![Robustness](robustness.png)

![48-hour response trace](response.png)

## Timing, verification and reproducibility

Median batch-one learned inference is approximately **{latency:.3f} ms** on this CPU; balance control is approximately {next(r['median_ms'] for r in result['timings'] if r['controller']=='balance'):.3f} ms. Timing uses 20 warmups and 200 measured calls per trained instance; it excludes sensors, networking, physics integration and actuators. A ten-minute simulated control interval is not a measured real-home reaction time. The source wiring supplies no demonstrated speed advantage. Masked dense recurrent tensors mean fewer active edges do not necessarily run faster.

Independent checks read all **48 initial/selected checkpoint files**, verified hashes and finite tensors, confirmed changes to active recurrent weights, and obtained identical actions after save/reload. All **180 primary held-out episodes** were replayed with exact recorded metrics on the training machine. A separate float64 NumPy implementation replayed four 48-hour traces (**1,152 steps**) and independently integrated temperature, CO₂, illumination, energy and deficits within declared tolerances. Simulator tests cover passive equilibrium, known CO₂ decay, heat capacity, energy integration, finite-difference derivatives, actuator bounds, reproducibility, graph degrees, hub isolation, surgery and API behavior. The initial evaluation hit an unclassified-cell metadata sorting error after scoring; only class-label handling was repaired, then the same frozen models/scenarios were reevaluated. A later review corrected functional-spoke lesions to silence the first message as well as subsequent updates; only validation interventions were recomputed. A regression test proves that a lesioned spoke cannot transmit its unique sensor input. The result metadata records the unchanged score hash. No trained weights, primary/stress scores, selection or reward were changed.

Run `uv sync --extra test`, `uv run --no-sync python -m pytest -q`, `uv run --no-sync python -m somahouse.verify`, then `uv run --no-sync python -m somahouse.app --port 8766`. Existing results are protected from training overwrite. To reproduce training in a fresh checkout, first move reports/ and checkpoints/ to an archive outside the checkout, recreate those directories, and run `python -m somahouse.study train` followed by `evaluate`, `verify` and `report` through uv. Preserve the included source seed and protocol. The exact environment is pinned in uv.lock. Different hardware/library builds may change training numerics.

The live lab runs inference with saved weights. Setpoint, occupancy, daylight, weather, sensor stress and hub/spoke interventions affect only an in-memory session. They are not additional research observations and do not update the trained model. The architecture inspector reads actual weights, source identifiers and surgery lineage. Downloadable artifacts include episode CSV, selection, checkpoints, protocol, verification and the study bundle. The portable docs/ case study works without a Python backend; the interactive house requires the local server.

## Limits and next experiment

This is a finite-budget proof of concept in one generator. Three seeds and 60 normal scenarios are limited. Synthetic schedules, fixed COP, simplified radiation and air mixing, no humidity, particulates, equipment delay, noisy communications, true weather records or human preference model restrict relevance. A high-order building emulator and a well-tuned PID/MPC reference are needed before external control claims. No hardware has been connected.

The study explores explicit training and structural operations, not open-ended self-evolution. One duplication and one salience pruning operation are tested per seed; no automatic invention of arbitrary wiring or lifelong adaptation is demonstrated. Circuit necessity is not established by a lesion alone. Future work should preregister a new held-out set, replace soft CO₂ and comfort penalties with constrained control, add sensor-fault handling, retrain without the weak context spoke, and evaluate on a calibrated BOPTEST/EnergyPlus building with matched sensor access. Online adaptation would then need a bounded rollback and independent constraints.

## Primary references and licenses

- [MaleCNS download, attribution and CC BY 4.0](https://male-cns.janelia.org/download/)
- [Google Research: male fruit fly connectome](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [EnergyPlus engineering zone heat-balance reference](https://energyplus.readthedocs.io/en/stable/guides/engineering-reference/2.1-basis-for-the-zone-and-air-system-integration.html)
- [BOPTEST core KPI definitions](https://ibpsa.github.io/project1-boptest/docs-design/core_kpi.html)
- [PODS: Policy Optimization via Differentiable Simulation](https://proceedings.mlr.press/v139/mora21a.html)

EnergyPlus and BOPTEST inform the physical balance and separate-metric methodology; this custom simulator has not been validated against either framework. Code is MIT; source/derived connectome data and trained checkpoints retain CC BY 4.0 attribution. Font licenses are bundled.
"""
    (REPORTS / "report.md").write_text(text)
    case = f"""# SomaHouse

## What if a house had a nervous system?

I wanted to see whether a small circuit derived from a fly connectome could learn to look after a house. Temperature, fresh air and lighting are different jobs, but they influence one another. Opening a vent changes the air and the heating load. Sunlight changes the lighting demand and the room temperature. That made a house an interesting way to test a modular brain.

I built a three-room simulator and gave the controller separate thermal, air, light and context circuits. They communicate through a central learned hub. The source wiring comes from actual MaleCNS neuron-pair synapse counts; the sensors, functional modules, hub and actuators are my engineering choices.

The question was not just whether it could learn. I also wanted to know what happened when I cut the circuit down, duplicated it, changed its wiring, or disconnected the hub.

## What I built

The simulator includes changing weather, daylight through windows, occupancy, temperature requests, limited heating and cooling, vents and dimmable lights. The controller sees current sensors and its last action. It learns from how its decisions affect the simulated house, using gradients through the physics. It does not see future weather or occupancy.

I trained 24 candidates across three random seeds, then froze selection before running the final tests. The study includes 5,850 optimizer updates and 3.37 million training transitions. I compared the learned circuits against conventional feedback and balance controllers, rewired and dense networks, a smaller circuit, and an equally trained base circuit. The simulator, trained weights, source IDs and full episode results are available with the project.

## The result I found useful

Duplication helped, but the gain was modest: **{improvement:.1f}% lower held-out cost** than giving the smaller circuit the same extra training. Cutting the circuit in half made performance worse. Removing the hub also hurt control, which shows the trained network was using communication between its modules.

The selected circuit had 256 spoke units, a 16-unit hub and 32 unique source neuron IDs reused across its modules. It kept occupied rooms within one degree of the requested temperature for **{learned['thermal_compliance']*100:.1f}%** of occupied time across the primary test scenarios.

## What did not win

The conventional balance controller still used less energy and met more comfort targets. The learned controller used **{energy_extra:.1f}% more electricity** than that reference. A dense network also outperformed the same-size fly-derived base. Those results do not support a claim that biological wiring is a better way to control a home.

The fault tests were revealing. Both controllers struggled badly when the temperature sensor was biased. A lower aggregate cost could even coincide with worse CO₂ compliance. That made it clear why I needed to report the individual measurements alongside the reward.

## What I learned

The interesting part was making the experiment inspectable: following signals through the hub, comparing structural changes against equal training, and being able to trace every copied cell back to its source. A larger circuit earned a small improvement here. That is useful evidence, but it is specific to this simulator and training budget.

This is a simulated control study, not a living brain or a system ready for someone's home. The next experiment would use a more detailed building model, explicit constraints on comfort and ventilation, and stronger fault handling. I would also test whether the weak context circuit can be removed after retraining.

Project by Michael Wong · 2026. Code MIT; MaleCNS-derived data and trained checkpoints CC BY 4.0. Full methods and limitations: reports/report.md.
"""
    (REPORTS / "case-study.md").write_text(case)
    article = ""
    for block in case.strip().split("\n\n"):
        escaped = html.escape(block)
        # This controlled copy uses paired emphasis only; no arbitrary HTML is accepted.
        parts = escaped.split("**")
        escaped = "".join(("<strong>" if i % 2 else "</strong>") + p if i else p for i,p in enumerate(parts))
        if block.startswith("# "):
            continue
        if block.startswith("## "):
            article += f"<h2>{escaped[3:]}</h2>\n"
        else:
            article += f"<p>{escaped}</p>\n"
    summary_table = "<div class='table-scroll'><table><thead><tr><th>Controller</th><th>Cost ↓</th><th>kWh/day ↓</th><th>Temperature</th><th>CO₂</th></tr></thead><tbody>"
    for name in ("balance", "duplicate", "continue", "cut", "feedback"):
        r=normal[name]
        summary_table += f"<tr class='{ 'primary-row' if name==primary else ''}'><td>{LABELS[name]}</td><td>{r['cost']:.4f}</td><td>{r['energy_kwh']/2:.2f}</td><td>{r['thermal_compliance']*100:.1f}%</td><td>{r['co2_compliance']*100:.1f}%</td></tr>"
    summary_table += "</tbody></table></div><p class='micro'>Compliance is the mean occupied-time fraction. Targets: ±1°C, CO₂ ≤1,000 ppm. Soft optimization targets; simulated evidence.</p>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="SomaHouse: I trained a fly-derived modular controller for a simulated house, then tested what duplication, pruning and a central hub changed."><title>SomaHouse — a homeostasis experiment by Michael Wong</title><link rel="stylesheet" href="/assets/style.css"></head><body><header><a class="brand" href="/">SomaHouse</a><nav aria-label="Main"><a href="/">House lab</a><a href="/brain.html">The circuit</a><a href="/case-study.html" aria-current="page">The study ↗</a></nav></header><main class="article"><p class="eyebrow">Michael Wong / independent research / 2026</p><h1>Teaching a circuit to<br>care for a house.</h1><p class="lede">An experiment in modular brains, home regulation and what actually improves when a circuit grows.</p><div class="evidence">24 trained candidates · 3 seeds · 60 normal test scenarios · 4 fault shifts · fully simulated</div>{article}
<figure><img src="/figures/comparison.png" alt="Chart comparing cost, energy and temperature discomfort across learned and conventional controllers"><figcaption>Conventional balance control remains a strong reference. Energy and comfort are shown separately.</figcaption></figure>{summary_table}
<h2>Did more circuit help?</h2><figure><img src="/figures/capacity.png" alt="Validation and held-out cost by circuit size with three training seeds"><figcaption>Duplication is compared with the same extra training. Salience pruning physically removes parameters.</figcaption></figure>
<h2>Learning and failure modes</h2><figure><img src="/figures/learning.png" alt="Validation learning curves across five base architectures"><figcaption>Shading shows the range across three training seeds; these are validation measurements.</figcaption></figure><figure><img src="/figures/robustness.png" alt="Heatmap of controller costs under severe weather, sensor bias, CO2 dropout and reduced capacity"><figcaption>Faults expose the limits of soft rewards and a simplified model. No safety guarantee is claimed.</figcaption></figure>
<h2>Read the full evidence</h2><p>The full report includes physical equations, reward weights, source provenance, per-seed results, confidence intervals, class lesions, exact checkpoint hashes, independent trajectory readback and limitations.</p><div class="downloads"><a href="/api/download/report/report.md">Full research report ↓</a><a href="/api/download/report/case-study.md">Portfolio copy ↓</a><a href="/api/download/report/episodes.csv">Every episode · CSV ↓</a><a href="/api/download/report/verification.json">Verification ↓</a><a href="/api/download/protocol/protocol.md">Frozen protocol ↓</a><a href="/api/download/report/study-bundle.zip">Study bundle ↓</a></div><p><a href="https://github.com/thekozugroup/SomaHouse">Code, trained circuits and reproducible study on GitHub ↗</a></p>
<h2>Sources</h2><ul class="sources"><li><a href="https://male-cns.janelia.org/download/">MaleCNS v1.0, FlyEM / HHMI Janelia, Cambridge, MRC LMB and Google Research · CC BY 4.0</a></li><li><a href="https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/">Google Research: the male fruit fly connectome</a></li><li><a href="https://energyplus.readthedocs.io/en/stable/guides/engineering-reference/2.1-basis-for-the-zone-and-air-system-integration.html">EnergyPlus zone heat-balance reference</a></li><li><a href="https://ibpsa.github.io/project1-boptest/docs-design/core_kpi.html">BOPTEST separate energy and comfort metrics</a></li><li><a href="https://proceedings.mlr.press/v139/mora21a.html">Policy Optimization via Differentiable Simulation</a></li></ul><p class="small">Independent reduced-order simulator. Not an official BOPTEST benchmark or a validated EnergyPlus building. Code MIT; derived connectome data and checkpoints CC BY 4.0.</p></main><footer><span>Michael Wong · SomaHouse · 2026</span><span>Measured results, including the negative ones.</span></footer></body></html>"""
    (ROOT / "somahouse/static/case-study.html").write_text(page)
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    (docs / "fonts").mkdir(exist_ok=True)
    for p in (ROOT / "somahouse/static/fonts").iterdir(): shutil.copy2(p, docs / "fonts" / p.name)
    css = (ROOT / "somahouse/static/style.css").read_text().replace("/assets/fonts/", "fonts/")
    (docs / "style.css").write_text(css)
    for p in REPORTS.glob("*.png"): shutil.copy2(p, docs / p.name)
    for name in ("report.md", "case-study.md", "episodes.csv", "verification.json", "selection.json", "results.json", "seed-results.csv"):
        shutil.copy2(REPORTS / name, docs / name)
    shutil.copy2(ROOT / "research/protocol.md", docs / "protocol.md")
    static_page = page.replace("/assets/style.css", "style.css").replace("/figures/", "").replace("/api/download/report/", "").replace("/api/download/protocol/", "")
    static_page = static_page.replace('<a class="brand" href="/">', '<a class="brand" href="index.html">')
    static_page = static_page.replace('<a href="/">House lab</a>', '<a href="https://github.com/thekozugroup/SomaHouse#run-the-lab">Run the lab</a>')
    static_page = static_page.replace('<a href="/brain.html">The circuit</a>', '<a href="report.md">Full report</a>').replace('href="/case-study.html"', 'href="index.html"')
    (docs / "index.html").write_text(static_page)
    write_json(docs / "project-entry.json", {"title": "SomaHouse", "author": "Michael Wong", "year": 2026,
                "description": "A fly-derived modular circuit learns to regulate a simulated house. Three-seed training, circuit surgery and transparent control comparisons.",
                "repository": "https://github.com/thekozugroup/SomaHouse", "caseStudy": "index.html",
                "tags": ["Machine learning", "Building control", "Connectomics", "Research"],
                "hero": "comparison.png", "status": "Simulation study completed"})
    # Bundle data and representative trained weights; omit itself to keep generation stable.
    with zipfile.ZipFile(REPORTS / "study-bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("README.md", "LICENSE", "THIRD_PARTY.md", "pyproject.toml", "uv.lock"):
            z.write(ROOT / name, name)
        for folder in ("somahouse", "tests"):
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and p.suffix in (".py", ".html", ".css", ".js", ".ttf", ".txt"):
                    z.write(p, str(p.relative_to(ROOT)))
        for name in ("report.md", "case-study.md", "episodes.csv", "selection.json", "results.json", "verification.json", "seed-results.csv"):
            z.write(REPORTS / name, f"reports/{name}")
        for p in sorted(REPORTS.glob("*.png")): z.write(p, f"reports/{p.name}")
        for p in sorted((ROOT / "checkpoints").glob("*.pt")): z.write(p, f"checkpoints/{p.name}")
        z.write(ROOT / "research/protocol.md", "research/protocol.md")
        for p in sorted((ROOT / "data/connectome").iterdir()): z.write(p, f"data/connectome/{p.name}")
    shutil.copy2(REPORTS / "study-bundle.zip", docs / "study-bundle.zip")
    manifest = {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                for folder in ("checkpoints", "data/connectome", "reports", "docs") for p in sorted((ROOT / folder).rglob("*"))
                if p.is_file() and p.suffix != ".log" and p.name != "manifest.json"}
    write_json(REPORTS / "manifest.json", manifest)
    print(json.dumps({"primary": primary, "duplication_improvement_pct": improvement,
                      "report_words": len(text.split()), "manifest_files": len(manifest), "verification": verification["passed"]}, indent=2))


if __name__ == "__main__": report()
