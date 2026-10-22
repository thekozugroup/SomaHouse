"""Report study 2 separately, including failed preregistered success gates."""
import html
import json

import matplotlib.pyplot as plt
import numpy as np

from .brain import ROOT
from .teacher import OUT, PRIMARY

LABELS = {"teacher": "Tuned conventional teacher", "balance": "Original balance controller",
          "fly-imitation": "Fly / imitation only", PRIMARY: "Fly / teacher + reward",
          "fly-reward-only": "Fly / reward only", "dense-imitation": "Dense / imitation only",
          "dense-teacher-reward": "Dense / teacher + reward"}
GATE_LABELS = {"cost_at_least_2pct_lower": "Cost at least 2% lower",
              "cost_ci_below_zero": "Cost advantage supported by the uncertainty interval",
              "all_training_seeds_lower": "Lower cost in all three training seeds",
              "energy_ci_within_2pct": "Electricity within the 2% tolerance",
              "thermal_compliance_ci_within_1pp": "Temperature compliance within one percentage point",
              "co2_compliance_ci_within_1pp": "CO₂ compliance within one percentage point",
              "light_compliance_ci_within_1pp": "Lighting compliance within one percentage point"}


def generate():
    selection = json.loads((OUT / "selection.json").read_text())
    result = json.loads((OUT / "results.json").read_text())
    normal = {r["controller"]: r for r in result["summaries"] if r["stress"] == "normal"}
    p, t = normal[PRIMARY], normal["teacher"]
    gain = (1 - p["cost"] / t["cost"]) * 100
    cost_phrase = f"{abs(gain):.2f}% {'lower' if gain >= 0 else 'higher'} cost"
    reward_gain = 100 * (1 - p["cost"] / normal["fly-reward-only"]["cost"])
    refine_gain = 100 * (1 - p["cost"] / normal["fly-imitation"]["cost"])
    energy_gain = 100 * (1 - p["energy_kwh"] / t["energy_kwh"])
    success = result["outperforms_teacher"]
    verdict = "The student passed every registered superiority gate." if success else "The student did not pass the registered superiority test."
    table = "|Controller|Cost ↓|kWh/day ↓|Temperature|CO₂|Light|\n|---|---:|---:|---:|---:|---:|\n"
    order = ["teacher", "balance", PRIMARY, "fly-imitation", "fly-reward-only", "dense-teacher-reward", "dense-imitation"]
    for name in order:
        r = normal[name]
        table += f"|{LABELS[name]}|{r['cost']:.5f}|{r['energy_kwh']/2:.3f}|{r['thermal_compliance']*100:.2f}%|{r['co2_compliance']*100:.2f}%|{r['light_compliance']*100:.2f}%|\n"
    gates = "\n".join(f"- {'PASS' if value else 'FAIL'} — {GATE_LABELS[key]}" for key, value in result["gates"].items())
    intervals = "|Metric: student minus teacher|Mean difference|Paired 95% interval|\n|---|---:|---|\n"
    for c in result["comparisons"][:5]:
        scale = 100 if "compliance" in c["metric"] else 1
        unit = " (percentage points)" if scale == 100 else ""
        intervals += f"|{c['metric']}{unit}|{c['difference']*scale:+.5f}|[{c['ci95'][0]*scale:+.5f}, {c['ci95'][1]*scale:+.5f}]|\n"
    total_updates = sum(r["updates"] for r in selection["runs"])
    transitions = sum(r.get("training_transitions", 0) + r.get("dataset_transitions", 0) for r in selection["runs"])
    pairs = sum(r.get("supervised_pairs_presented", 0) for r in selection["runs"])
    seconds = sum(r["training_seconds"] for r in selection["runs"])
    text = f"""# Study 2 — can a teacher help the circuit surpass conventional control?

{verdict} The student has {cost_phrase} than the tuned teacher and uses {energy_gain:.2f}% less electricity. It meets fewer occupied temperature and CO₂ targets. Teaching plus refinement lowers cost {reward_gain:.2f}% versus reward-only training and {refine_gain:.2f}% versus imitation alone. These are new models and new test scenarios; study-1 scores remain unchanged.

## Why I added this experiment

The first experiment learned useful control but did not beat its strongest conventional reference. I wanted to see whether teaching the circuit good actions first, then letting it optimize their consequences, could close that gap. I also made the conventional reference stronger. Beating the weaker old reference would not answer whether the student can surpass the teacher it learned from.

The plan in research/teacher-protocol.md was committed before training (89a8294). The primary family was fixed as fly teacher + reward. Teacher gains, checkpoint hashes and median-validation representative seed were sealed in commit 9bfad28 before opening final tests. No hyperparameters, scored cost weights or checkpoints were changed after those tests. Original report scores and training artifacts were preserved.

## Teacher, student and information access

The conventional teacher is the lowest validation-cost controller among 36 nominal heat/air-balance gain combinations and the original balance baseline. The grid tests thermal gain 0.65/1.3/2.0, occupied offset 0/0.4/0.75°C toward outdoors, CO₂ target 850/950 ppm and ventilation feedback factor 0.6/1.2. Selected parameters: `{json.dumps(selection['teacher']['config'], sort_keys=True)}`. Validation cost: {selection['teacher']['validation']['cost']:.5f}. It uses fixed nominal building constants, current sensors and previous actions, no future schedules or true randomized plant parameters. This is a limited tuned conventional family, not an optimized MPC benchmark.

Every neural family uses eight 32-cell private spokes and the same learned 16-unit hub: 256 spoke units plus the hub. Fly masks retain 32 unique MaleCNS source identifiers; copies increase artificial units, not source diversity. Dense controls have the same allocated tensor dimensions with more active recurrent edges. All neural outputs come from the saved standalone network. The teacher is absent at inference; no thermostat overlay or teacher correction rescues neural actions. The source topology, artificial hub, engineered sensor assignments and tanh units have the same biological-fidelity limits as study 1.

## Teaching and reward feedback

For seeds 107, 119 and 141, collect 1,024 training episodes of 24 steps under the teacher, then train on 512 observation/action pairs per update for 400 Adam updates at .003. The target is the teacher's executed actuator command. Supervised loss is mean squared action error with light components weighted four. At update 200, add 256 twelve-step student-driven episodes with teacher action labels. This counters the shift between teacher and student states. It resembles teaching on student-visited states, but is not presented as a full DAgger implementation.

The selected imitation checkpoint starts 800 reward updates at Adam .001. Each update rolls 32 independent eight-hour windows, 48 ten-minute decisions, with direct gradients through the simulator. Gradient norms are clipped to five. The original scored cost is retained, with two additional declared training regularizers: occupied CO₂ excess above 950 ppm, weight 0.5 on normalized squared excess, and occupied light deficit below 300 lux, weight 2. These soft regularizers do not guarantee constraint satisfaction. Validation and test selection use the original cost only.

Controls retain the imitation-only checkpoint, train the same initial fly model with 1,200 reward-only updates, and repeat imitation + reward with dense wiring. Fly/dense imitation has 400 supervised updates; refinement has 800 simulated updates; reward-only has 1,200 simulated updates. Equal optimizer counts do not mean equal compute or interaction budget: supervised steps are cheaper and reward-only observes more transitions. Comparing final teacher + reward to its own imitation parent isolates refinement; comparing fly versus dense helps check whether source wiring is responsible.

Total: {len(selection['runs'])} selected stage checkpoints, six initial checkpoints, {total_updates:,} optimizer updates, {pairs:,} supervised pair presentations, and {transitions:,} training/data-collection transitions. Reward optimization contributes {sum(r.get('training_transitions',0) for r in selection['runs']):,} transitions. Summed recorded training time is {seconds/60:.2f} minutes on the same M1 Max CPU, two PyTorch threads; initial teacher-dataset collection and conventional grid search are outside stage timing. Validation occurs at step zero and every 100 updates, retaining the lowest original validation cost including step zero.

## Fresh evaluation and uncertainty

Training scenario IDs are 5,000,000–5,999,999. Validation uses 32 independent 48-hour IDs 6,000,001–6,000,032; final test uses 100 IDs 7,000,001–7,000,100. Four fault shifts each reuse 24 IDs 8,000,001–8,000,024 for pairing. No ID overlaps study 1's validation or final sets. These are new samples from the same synthetic generator, not a new building simulator or external validation dataset.

{result['episodes']:,} evaluated policy episodes / {result['transitions']:,} transitions arise from 17 policy instances and 196 scenario/shift pairs; only 124 unique exogenous seed IDs exist. Neural family means average three training seeds. Conventional controls run once per scenario. Paired intervals use 5,000 resamples of 100 normal scenario IDs, RNG seed 1907, after averaging training seeds. Retraining uncertainty and multiple-comparison correction are absent; secondary comparisons are descriptive.

{table}

Daily energy divides 48-hour totals by two. Compliance is the mean occupied-time fraction per scenario: temperature within ±1°C, CO₂ ≤1,000 ppm, light ≥295 lux. Full occupied deficit integrals, action movement and peak power are in episodes.csv. All controllers receive identical current observations and actuator constraints. Aggregate cost cannot alone demonstrate safe or uniformly better control.

![Fresh-test comparison](comparison.png)

## Registered superiority gates

The primary claim requires at least 2% lower mean cost; paired cost interval below zero; lower cost in all three training seeds; lower paired confidence bound on each compliance difference above -1 percentage point; and upper paired energy-increase bound below 2% of teacher mean energy. Every condition must pass. These are practical non-inferiority margins selected before training, not post-hoc excuses for a lower reward.

{gates}

{intervals}

Individual primary costs: {', '.join(f"seed {r['seed']}: {r['cost']:.5f}" for r in result['primary_per_seed'])}. Full trajectories and dense/reward-only controls are retained even when they weaken a preferred explanation. A win over the original balance baseline does not establish a win over the tuned teacher. A dense model matching or beating the fly model does not establish a connectome advantage.

![Validation training curves](learning.png)

## Stress results

|Fault|Teacher cost|Fly teacher + reward cost|Teacher temperature / CO₂|Fly temperature / CO₂|
|---|---:|---:|---|---|
"""
    for stress in ("weather", "temperature_bias", "co2_dropout", "capacity"):
        a = next(r for r in result["summaries"] if r["controller"] == "teacher" and r["stress"] == stress)
        b = next(r for r in result["summaries"] if r["controller"] == PRIMARY and r["stress"] == stress)
        text += f"|{stress}|{a['cost']:.4f}|{b['cost']:.4f}|{a['thermal_compliance']*100:.1f}% / {a['co2_compliance']*100:.1f}%|{b['thermal_compliance']*100:.1f}% / {b['co2_compliance']*100:.1f}%|\n"
    text += """
Stress results are descriptive and do not select checkpoints. Sensor bias, dropout and capacity limits can defeat both designs. Simulation reward refinement is not a substitute for redundant sensors, calibrated plant identification, independent limits or real-building trials. No physical house was controlled and no claim of safe deployment is made.

## Readback, reproducibility and next question

Run `python -m somahouse.teacher_verify` to verify selected checkpoint hashes, exact save/reload actions, finite tensors, changes to active recurrent weights, all 1,700 normal controller episodes, comparison intervals and two independently integrated NumPy traces. `--portable` permits the same unit-aware CPU rounding tolerances as study 1. See verification.json for actual outcomes. The live inspector can load all saved study-2 models. Page interaction and video playback never update their weights.

For reproduction in a separate checkout, archive reports/teacher and checkpoints/teacher outside that checkout; run `python -m somahouse.teacher train`, commit the sealed selection, then `evaluate`, `somahouse.teacher_verify` and `somahouse.report`. Completed evaluation is protected from overwrite. The original physics, source seed, versions and scored reward remain pinned. Hardware differences can change training numerics.

The useful next question is whether a controller with better identification of the house and explicit constraints can improve the energy/comfort tradeoff on a calibrated external building model. More copies alone are not evidence of better control. Teacher-guided training can be assessed without claiming that the artificial fly circuit is biologically intelligent or universally better than conventional engineering.

Method references: [Policy Optimization with Demonstrations](https://proceedings.mlr.press/v80/kang18a.html), [Policy Optimization via Differentiable Simulation](https://proceedings.mlr.press/v139/mora21a.html). These provide methodological context; this implementation is independent supervised imitation plus differentiable policy optimization, not model-free RL or either published algorithm. MaleCNS attribution and CC BY 4.0 carry over to derived data and checkpoints; original code/report remain MIT.
"""
    verification_file = OUT / "verification.json"
    text += "\n\n## Secondary cost comparisons\n\nStudent minus reference; negative favors the teacher-guided fly model. These comparisons were declared, but are secondary and not corrected for multiple testing.\n\n|Reference|Cost difference|Paired 95% interval|\n|---|---:|---|\n"
    for c in result["comparisons"][5:]:
        text += f"|{LABELS[c['b']]}|{c['difference']:+.6f}|[{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]|\n"
    if verification_file.exists():
        checks = json.loads(verification_file.read_text())
        errors = checks["validation_teacher_state_action_mse"]
        text += "\n\n## Supervision diagnostic\n\nUnweighted executed-action mean squared error on the 32 validation teacher trajectories. This evaluates copying on teacher-visited states, not closed-loop performance, and was not used to change selected checkpoints.\n\n|Family|Seed 107|Seed 119|Seed 141|\n|---|---:|---:|---:|\n"
        for family in ("fly-imitation", PRIMARY, "fly-reward-only", "dense-imitation", "dense-teacher-reward"):
            text += f"|{LABELS[family]}|" + "|".join(f"{errors[f'{family}-s{seed}']:.6f}" for seed in (107, 119, 141)) + "|\n"
        text += f"\nReadback passed: {checks['selected_checkpoints']} selected models, {checks['initial_checkpoints']} initial models, {sum(r['episodes'] for r in checks['normal_test_replay']):,} normal policy episodes, {sum(r['steps'] for r in checks['numpy_physics'])} independent NumPy physical steps, and all {checks['comparisons_recomputed']} comparison intervals recomputed from episode CSV. Mode: {checks['mode']}.\n"
    (OUT / "report.md").write_text(text)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, ("cost", "energy_kwh"), ("Original weighted cost ↓", "Electricity · kWh/day ↓")):
        vals = [normal[n][key] / (2 if key == "energy_kwh" else 1) for n in order]
        ax.barh(range(len(order)), vals, color=["#252525" if n == PRIMARY else "#777777" if n == "teacher" else "#b5b5b5" for n in order])
        ax.set_yticks(range(len(order)), [LABELS[n] for n in order], fontsize=8); ax.invert_yaxis()
        ax.set_xlabel(title); ax.grid(axis="x", alpha=.4); ax.set_axisbelow(True)
    fig.suptitle("Study 2 / fresh held-out scenarios / teacher gains tuned on validation")
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for index, family in enumerate((PRIMARY, "fly-reward-only", "dense-teacher-reward")):
        runs = [r for r in selection["runs"] if r["family"] == family]
        values = np.array([[v["validation"]["cost"] for v in r["curve"]] for r in runs])
        x = [v["update"] for v in runs[0]["curve"]]
        color = ("#222222", "#777777", "#aaaaaa")[index]
        ax.plot(x, values.mean(0), color=color, ls=("-", "--", ":")[index], label=LABELS[family])
        ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.1)
    ax.axhline(selection["teacher"]["validation"]["cost"], color="#555555", ls="-.", label="Tuned teacher")
    ax.set(xlabel="Reward updates / teacher-guided models start after imitation", ylabel="Validation cost ↓", ylim=(.08, .25))
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Learning after teaching / range across three training seeds / costs above .25 clipped")
    fig.tight_layout(); fig.savefig(OUT / "learning.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    table_html = '<div class="table-scroll"><table><thead><tr><th>Controller</th><th>Cost ↓</th><th>kWh/day ↓</th><th>Temperature</th><th>CO₂</th><th>Light</th></tr></thead><tbody>'
    for name in order:
        r = normal[name]
        table_html += f'<tr><td>{LABELS[name]}</td><td>{r["cost"]:.5f}</td><td>{r["energy_kwh"]/2:.2f}</td><td>{r["thermal_compliance"]*100:.1f}%</td><td>{r["co2_compliance"]*100:.1f}%</td><td>{r["light_compliance"]*100:.1f}%</td></tr>'
    table_html += '</tbody></table></div>'
    section = f'''<section id="teacher-study"><h2>Study 2 / can a controller teach the circuit?</h2><p>I taught a new circuit to copy a conventional controller, then let it improve through reward feedback. I also tuned the teacher and reserved 100 fresh test scenarios. {verdict} The original study remains below, with its scores preserved.</p><p>Teaching plus refinement reduced cost <strong>{reward_gain:.1f}% versus reward-only training</strong> and {refine_gain:.1f}% versus imitation alone. Against the tuned teacher, the brain had <strong>{cost_phrase}</strong> and used {energy_gain:.1f}% less electricity, while meeting fewer temperature and air-quality targets. That tradeoff matters more than a lower electricity number alone.</p>{table_html}<figure><img src="/figures/teacher/comparison.png" alt="Fresh-test cost and energy for imitation, reward-only, teacher-guided and conventional controls"><figcaption>Study 2 uses new test scenarios. A lower cost alone does not establish superiority.</figcaption></figure><ul>{''.join(f'<li>{"Pass" if v else "Fail"}: {html.escape(GATE_LABELS[k])}</li>' for k,v in result['gates'].items())}</ul><div class="downloads"><a href="/api/download/teacher/report.md">Full teacher-study report ↓</a><a href="/api/download/teacher/episodes.csv">Every new episode ↓</a><a href="/api/download/teacher/teacher-protocol.md">Registered plan ↓</a><a href="/api/download/teacher/verification.json">Study 2 verification ↓</a></div></section>'''
    return text, section
