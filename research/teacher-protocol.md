# SomaHouse — study 2: teaching before reward training

Registered before study-2 training. Study 1 remains immutable. Its published
results motivate this extension; none of its final seeds are reused here.

Question: can a standalone fly-derived policy first imitate a conventional
controller, then improve on it through simulator reward feedback? A positive
result would concern this generator and these controllers, not all conventional
control or a special advantage of biological wiring.

Keep the study-1 physics, observations, cost weights, actuator limits, 32-cell
source subset, engineered 16-unit hub, two copies per functional domain and all
source attribution unchanged. No conventional controller runs underneath the
student at inference. No new circuit growth is introduced in this extension.

## Data and comparison

Training seeds 107, 119, 141; scenario training range 5,000,000–5,999,999.
Validation: 6,000,001–6,000,032, 48 hours. Final test: 7,000,001–7,000,100,
48 hours. Stress: 8,000,001–8,000,024, each used for weather, temperature bias,
CO2 dropout and reduced capacity. Disjoint from study 1 and one another.

Tune a conventional nominal balance controller on validation only: thermal gain
{0.65, 1.3, 2.0}, occupied temperature offset toward outdoor temperature
{0, 0.4, 0.75} degC, CO2 target {850, 950} ppm, ventilation feedback multiplier
{0.6, 1.2}: 36 combinations, plus the original balance controller. The lowest
validation-cost member is the teacher and strongest conventional reference.
Same current sensors, no forecasts or randomized true plant constants.

Three independent retraining seeds for each neural family:

- fly imitation: 400 supervised updates; lowest validation-cost checkpoint.
- fly teacher + reward: start from that selected imitation checkpoint, then
  800 reward updates; retain step zero if reward refinement worsens validation.
- fly reward only: 1,200 reward updates from the exact same initial fly weights.
- dense teacher + reward: same widths/copies, imitation and refinement schedule,
  with dense recurrent wiring; report its imitation checkpoint too.

Imitation uses 512 observation/action pairs per Adam update, learning rate .003,
gradient norm clipping 5. Build observations from 1,024 training episodes of
24 steps under the teacher. Add 256 episodes of 12 steps under the student at
update 200, labeled by the teacher; detach all collected states and labels.
This is teacher supervision on student-visited states, not a full DAgger claim.
Loss is mean squared executed-action error, with lighting errors weighted 4.

Reward refinement uses 32 parallel training windows, 48 steps each (8 hours),
Adam .001, gradient clipping 5. Original scored simulator cost is unchanged.
Training also penalizes occupied CO2 above 950 ppm (weight 0.5 on normalized
excess squared) and light below 300 lux (weight 2 on normalized deficit squared).
These declared training regularizers discourage reward tradeoffs; they are
not hard constraints. Only original cost selects checkpoints and scores tests.
Reward-only uses the identical objective and 1,200 updates; datasets/draws and
different supervised versus simulated-compute budgets are recorded explicitly.

Validate at step zero, every 100 updates and the last update. Freeze teacher,
selected checkpoint hashes and the primary family (fly teacher + reward)
before opening final tests. Choose a representative training seed by median
validation rank, scenario fixed to 7,000,001, irrespective of test performance.
Never extend training or tune hyperparameters based on final results.

## Success gate and reporting

Primary: fly teacher + reward versus tuned teacher. A claim of outperforming
requires at least 2% lower mean original weighted cost, a paired 95% bootstrap
cost interval wholly below zero, and mean cost lower in all three training seeds.
Also require paired lower confidence bounds on occupied temperature, CO2 and
light compliance differences above -1 percentage point, and paired upper bound
on electricity increase below 2% of teacher mean energy. Report any failed gate.
Use 5,000 paired scenario resamples, RNG seed 1907, after averaging training
seeds. Publish per-seed results; this interval excludes retraining uncertainty.

Report imitation error, all independent physical metrics, learning curves,
all 100 primary scenarios, all stress results, source/parameter counts,
training budgets and timing, fixed-weight save/reload and trajectory readback.
Stress is descriptive, not another selection stage. Secondary comparisons
have no multiple-testing correction and cannot establish universal superiority.
Show study 1 and study 2 separately. Do not rewrite the original negative result.

Method background: demonstration-guided policy learning
https://proceedings.mlr.press/v80/kang18a.html and differentiable policy learning
https://proceedings.mlr.press/v139/mora21a.html. This is an independent simple
implementation using supervised imitation and direct simulator gradients,
not those algorithms, model-free RL, or online learning from the website.
