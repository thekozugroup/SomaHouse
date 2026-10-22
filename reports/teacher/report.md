# Study 2 — can a teacher help the circuit surpass conventional control?

The student did not pass the registered superiority test. The student has 1.34% higher cost than the tuned teacher and uses 8.76% less electricity. It meets fewer occupied temperature and CO₂ targets. Teaching plus refinement lowers cost 7.45% versus reward-only training and 10.05% versus imitation alone. These are new models and new test scenarios; study-1 scores remain unchanged.

## Why I added this experiment

The first experiment learned useful control but did not beat its strongest conventional reference. I wanted to see whether teaching the circuit good actions first, then letting it optimize their consequences, could close that gap. I also made the conventional reference stronger. Beating the weaker old reference would not answer whether the student can surpass the teacher it learned from.

The plan in research/teacher-protocol.md was committed before training (89a8294). The primary family was fixed as fly teacher + reward. Teacher gains, checkpoint hashes and median-validation representative seed were sealed in commit 9bfad28 before opening final tests. No hyperparameters, scored cost weights or checkpoints were changed after those tests. Original report scores and training artifacts were preserved.

## Teacher, student and information access

The conventional teacher is the lowest validation-cost controller among 36 nominal heat/air-balance gain combinations and the original balance baseline. The grid tests thermal gain 0.65/1.3/2.0, occupied offset 0/0.4/0.75°C toward outdoors, CO₂ target 850/950 ppm and ventilation feedback factor 0.6/1.2. Selected parameters: `{"co2_target": 950.0, "gain": 2.0, "offset": 0.0, "vent_gain": 1.2}`. Validation cost: 0.09353. It uses fixed nominal building constants, current sensors and previous actions, no future schedules or true randomized plant parameters. This is a limited tuned conventional family, not an optimized MPC benchmark.

Every neural family uses eight 32-cell private spokes and the same learned 16-unit hub: 256 spoke units plus the hub. Fly masks retain 32 unique MaleCNS source identifiers; copies increase artificial units, not source diversity. Dense controls have the same allocated tensor dimensions with more active recurrent edges. All neural outputs come from the saved standalone network. The teacher is absent at inference; no thermostat overlay or teacher correction rescues neural actions. The source topology, artificial hub, engineered sensor assignments and tanh units have the same biological-fidelity limits as study 1.

## Teaching and reward feedback

For seeds 107, 119 and 141, collect 1,024 training episodes of 24 steps under the teacher, then train on 512 observation/action pairs per update for 400 Adam updates at .003. The target is the teacher's executed actuator command. Supervised loss is mean squared action error with light components weighted four. At update 200, add 256 twelve-step student-driven episodes with teacher action labels. This counters the shift between teacher and student states. It resembles teaching on student-visited states, but is not presented as a full DAgger implementation.

The selected imitation checkpoint starts 800 reward updates at Adam .001. Each update rolls 32 independent eight-hour windows, 48 ten-minute decisions, with direct gradients through the simulator. Gradient norms are clipped to five. The original scored cost is retained, with two additional declared training regularizers: occupied CO₂ excess above 950 ppm, weight 0.5 on normalized squared excess, and occupied light deficit below 300 lux, weight 2. These soft regularizers do not guarantee constraint satisfaction. Validation and test selection use the original cost only.

Controls retain the imitation-only checkpoint, train the same initial fly model with 1,200 reward-only updates, and repeat imitation + reward with dense wiring. Fly/dense imitation has 400 supervised updates; refinement has 800 simulated updates; reward-only has 1,200 simulated updates. Equal optimizer counts do not mean equal compute or interaction budget: supervised steps are cheaper and reward-only observes more transitions. Comparing final teacher + reward to its own imitation parent isolates refinement; comparing fly versus dense helps check whether source wiring is responsible.

Total: 15 selected stage checkpoints, six initial checkpoints, 10,800 optimizer updates, 1,228,800 supervised pair presentations, and 13,068,288 training/data-collection transitions. Reward optimization contributes 12,902,400 transitions. Summed recorded training time is 14.77 minutes on the same M1 Max CPU, two PyTorch threads; initial teacher-dataset collection and conventional grid search are outside stage timing. Validation occurs at step zero and every 100 updates, retaining the lowest original validation cost including step zero.

## Fresh evaluation and uncertainty

Training scenario IDs are 5,000,000–5,999,999. Validation uses 32 independent 48-hour IDs 6,000,001–6,000,032; final test uses 100 IDs 7,000,001–7,000,100. Four fault shifts each reuse 24 IDs 8,000,001–8,000,024 for pairing. No ID overlaps study 1's validation or final sets. These are new samples from the same synthetic generator, not a new building simulator or external validation dataset.

3,332 evaluated policy episodes / 959,616 transitions arise from 17 policy instances and 196 scenario/shift pairs; only 124 unique exogenous seed IDs exist. Neural family means average three training seeds. Conventional controls run once per scenario. Paired intervals use 5,000 resamples of 100 normal scenario IDs, RNG seed 1907, after averaging training seeds. Retraining uncertainty and multiple-comparison correction are absent; secondary comparisons are descriptive.

|Controller|Cost ↓|kWh/day ↓|Temperature|CO₂|Light|
|---|---:|---:|---:|---:|---:|
|Tuned conventional teacher|0.09052|22.150|98.06%|99.83%|100.00%|
|Original balance controller|0.09469|22.005|96.86%|99.72%|100.00%|
|Fly / teacher + reward|0.09173|20.210|96.08%|95.47%|99.02%|
|Fly / imitation only|0.10199|22.467|96.34%|84.55%|78.84%|
|Fly / reward only|0.09911|23.380|96.30%|96.09%|98.42%|
|Dense / teacher + reward|0.09243|20.584|95.98%|96.70%|98.72%|
|Dense / imitation only|0.10110|22.517|96.47%|86.48%|82.47%|


Daily energy divides 48-hour totals by two. Compliance is the mean occupied-time fraction per scenario: temperature within ±1°C, CO₂ ≤1,000 ppm, light ≥295 lux. Full occupied deficit integrals, action movement and peak power are in episodes.csv. All controllers receive identical current observations and actuator constraints. Aggregate cost cannot alone demonstrate safe or uniformly better control.

![Fresh-test comparison](comparison.png)

## Registered superiority gates

The primary claim requires at least 2% lower mean cost; paired cost interval below zero; lower cost in all three training seeds; lower paired confidence bound on each compliance difference above -1 percentage point; and upper paired energy-increase bound below 2% of teacher mean energy. Every condition must pass. These are practical non-inferiority margins selected before training, not post-hoc excuses for a lower reward.

- FAIL — Cost at least 2% lower
- FAIL — Cost advantage supported by the uncertainty interval
- FAIL — Lower cost in all three training seeds
- PASS — Electricity within the 2% tolerance
- FAIL — Temperature compliance within one percentage point
- FAIL — CO₂ compliance within one percentage point
- FAIL — Lighting compliance within one percentage point

|Metric: student minus teacher|Mean difference|Paired 95% interval|
|---|---:|---|
|cost|+0.00121|[-0.00197, +0.00497]|
|energy_kwh|-3.88046|[-4.22267, -3.56115]|
|thermal_compliance (percentage points)|-1.97673|[-2.60133, -1.40011]|
|co2_compliance (percentage points)|-4.35374|[-5.77986, -3.06560]|
|light_compliance (percentage points)|-0.98326|[-1.27014, -0.73040]|


Individual primary costs: seed 107: 0.09144, seed 119: 0.09221, seed 141: 0.09155. Full trajectories and dense/reward-only controls are retained even when they weaken a preferred explanation. A win over the original balance baseline does not establish a win over the tuned teacher. A dense model matching or beating the fly model does not establish a connectome advantage.

![Validation training curves](learning.png)

## Stress results

|Fault|Teacher cost|Fly teacher + reward cost|Teacher temperature / CO₂|Fly temperature / CO₂|
|---|---:|---:|---|---|
|weather|0.6871|0.8048|86.1% / 99.9%|76.7% / 88.0%|
|temperature_bias|1.1618|1.3070|2.9% / 99.9%|4.1% / 96.8%|
|co2_dropout|2.5935|0.5286|98.5% / 15.9%|97.0% / 26.2%|
|capacity|5.0532|4.7175|48.0% / 29.9%|43.7% / 27.5%|

Stress results are descriptive and do not select checkpoints. Sensor bias, dropout and capacity limits can defeat both designs. Simulation reward refinement is not a substitute for redundant sensors, calibrated plant identification, independent limits or real-building trials. No physical house was controlled and no claim of safe deployment is made.

## Readback, reproducibility and next question

Run `python -m somahouse.teacher_verify` to verify selected checkpoint hashes, exact save/reload actions, finite tensors, changes to active recurrent weights, all 1,700 normal controller episodes, comparison intervals and two independently integrated NumPy traces. `--portable` permits the same unit-aware CPU rounding tolerances as study 1. See verification.json for actual outcomes. The live inspector can load all saved study-2 models. Page interaction and video playback never update their weights.

For reproduction in a separate checkout, archive reports/teacher and checkpoints/teacher outside that checkout; run `python -m somahouse.teacher train`, commit the sealed selection, then `evaluate`, `somahouse.teacher_verify` and `somahouse.report`. Completed evaluation is protected from overwrite. The original physics, source seed, versions and scored reward remain pinned. Hardware differences can change training numerics.

The useful next question is whether a controller with better identification of the house and explicit constraints can improve the energy/comfort tradeoff on a calibrated external building model. More copies alone are not evidence of better control. Teacher-guided training can be assessed without claiming that the artificial fly circuit is biologically intelligent or universally better than conventional engineering.

Method references: [Policy Optimization with Demonstrations](https://proceedings.mlr.press/v80/kang18a.html), [Policy Optimization via Differentiable Simulation](https://proceedings.mlr.press/v139/mora21a.html). These provide methodological context; this implementation is independent supervised imitation plus differentiable policy optimization, not model-free RL or either published algorithm. MaleCNS attribution and CC BY 4.0 carry over to derived data and checkpoints; original code/report remain MIT.


## Secondary cost comparisons

Student minus reference; negative favors the teacher-guided fly model. These comparisons were declared, but are secondary and not corrected for multiple testing.

|Reference|Cost difference|Paired 95% interval|
|---|---:|---|
|Original balance controller|-0.002959|[-0.006720, +0.000978]|
|Fly / imitation only|-0.010254|[-0.013947, -0.006465]|
|Fly / reward only|-0.007383|[-0.010274, -0.003905]|
|Dense / teacher + reward|-0.000701|[-0.001568, +0.000350]|


## Supervision diagnostic

Unweighted executed-action mean squared error on the 32 validation teacher trajectories. This evaluates copying on teacher-visited states, not closed-loop performance, and was not used to change selected checkpoints.

|Family|Seed 107|Seed 119|Seed 141|
|---|---:|---:|---:|
|Fly / imitation only|0.001269|0.001298|0.001259|
|Fly / teacher + reward|0.023766|0.023345|0.019992|
|Fly / reward only|0.034807|0.053410|0.036526|
|Dense / imitation only|0.001229|0.001297|0.001029|
|Dense / teacher + reward|0.023200|0.024664|0.006322|

Readback passed: 15 selected models, 6 initial models, 1,700 normal policy episodes, 576 independent NumPy physical steps, and all 9 comparison intervals recomputed from episode CSV. Mode: strict_1e-7.
