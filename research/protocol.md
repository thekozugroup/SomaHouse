# SomaHouse — study 1, frozen plan

Registered before training and before final evaluation, 22 October 2026.

## Question

Can a small artificial network initialized from actual fly synapse counts learn to regulate a three-room home? Does an engineered central communication hub help? Does adding circuit capacity help, or can we retain a smaller useful subset?

This is a custom reduced-order simulation study. It is not an EnergyPlus or BOPTEST benchmark, a reconstructed living brain, a real-home trial, or evidence of consciousness. The hub and functional assignment are engineered. A fly has no mammalian cortex. Biological origin alone establishes neither speed nor control quality.

## House and controls

Three rooms: living, bedroom, office. Zone heat and CO2 mass balances, outdoor weather, solar/daylight, occupancy schedules and unannounced visits, setpoint changes, ventilation heat exchange, finite heating/cooling capacity, dimmable lights, electrical energy and action changes. Ten-minute control steps. CO2 is a ventilation proxy; the study's 1,000 ppm target is an engineering assumption, not a universal health boundary. Humidity, particulate chemistry, envelope moisture and equipment cycling are outside this first model.

Every controller receives the same current sensor observations and previous executed actions. No future weather or occupancy is supplied to the neural policy. The feedforward balance baseline has nominal plant constants, as explicitly disclosed; it does not receive the randomized true building parameters. Weather and plant parameters vary between scenarios. All controllers share actuator constraints. No hidden thermostat rescues neural actions.

## Brain and training

Four artificial spokes: thermal, air, light and context. Each sees a different explicit sensor mask. Fly-derived recurrent edges are masked by actual MaleCNS connectivity. A learned 16-unit hub collects 8-dimensional messages from every spoke and broadcasts to all spokes twice per decision. Spokes have private inputs, weights and output heads. There are no direct spoke-to-spoke edges. Source transmitter labels are initialization heuristics, not electrophysiology.

Use direct policy optimization through differentiable simulation: gradients of multi-step energy/comfort costs train the controller. This is model-based learning with access to simulator derivatives, not model-free PPO, online home learning, or a proprietary language model. Train from random adapter weights without imitation targets. Learning windows start at randomized times and states; evaluate complete episodes separately.

Training seeds: 7, 19, 41. Baseline architectures: fly32, fly64, degree-preserving rewired32, dense32, nohub32. Here 32/64 is source neurons per spoke, not total system size. Same optimization schedule and exogenous training draws per seed. Compare the initially untrained fly32 as well.

Surgery descendants of each trained fly32: equal-budget continue; cut to 16 source cells per spoke selected by learned weight salience; duplicate every spoke with initially split readouts and hub aggregation. Fine-tune all three for equal extra updates, retain step zero if validation worsens. Duplication is an engineering copy, not new biological neurons. No claim that salience proves anatomical necessity.

Optimization budget after a training-only speed/gradient smoke: 300 base updates, 150 surgery updates, 24 parallel scenarios, 24 ten-minute steps per update. Adam, gradient clipping. Validation every 50 updates, plus update zero and final. Select the lowest validation cost checkpoint. A failed finite-gradient check stops the run. A schedule change must be recorded before the final test is opened.

## Selection and held-out evaluation

Training scenario seeds 1,000–1,999,999; validation 2,000,001–2,000,024. Primary test seeds 3,000,001–3,000,060, each 48 hours. No shared scenario seed across splits. Stress seeds 4,000,001–4,000,016, each 48 hours, for each of four declared shifts: severe weather, biased temperature sensor, CO2 sensor dropout, reduced actuator capacity. Stress results are separate from primary selection.

Conventional baselines: proportional feedback (thermostat, demand ventilation, daylight dimming); nominal heat/mass-balance feedforward with feedback; off/minimum operation. Baseline gains and neural cost coefficients are fixed before any test rollout.

Publish total kWh; occupied degree-hours outside ±1°C of requested temperature; occupied CO2 excess ppm-hours above 1,000 ppm; occupied light deficit lux-hours below 300 lux; occupied target compliance rates; actuator movement; training objective; inference timing; trainable/allocated parameter counts; source and artificial neuron counts. Report day-normalized values for readability. Objective weights and units must be published. No energy savings claim without the accompanying comfort tradeoff.

Selection uses validation objective only. Among fly-derived candidates within 2% of the best validation objective, choose the fewest source/copy neurons, then lowest objective. Evaluate all candidates on test once to expose seed variance and capacity tradeoffs, without changing the primary selected family or checkpoint. A duplication benefit requires lower validation cost across all three seeds and a held-out paired 95% bootstrap interval excluding zero; otherwise keep the smaller candidate. Compare rewired and dense controls without claiming that a non-significant difference proves equivalence.

Primary uncertainty: paired bootstrap over independent scenario seeds after averaging the three training seeds, 2,000 resamples, deterministic bootstrap seed 918. Also publish individual training-seed results. This interval measures scenario variability within this generator, not real-building uncertainty; three training seeds are limited.

Post-hoc interventions: hub messages zeroed; each functional spoke zeroed; recurrence zeroed; source-cell class lesions. Evaluate on validation, not as another final-test selection stage. These establish computational sensitivity, not biological function. Lesions are not retrained.

## Verification and deliverables

Deterministic simulator tests, passive equilibrium and CO2 balance checks, correct energy integration, finite gradients through physics and policy, graph degree preservation, module isolation/no-hub tests, surgery output-preservation checks, serialization readback, split disjointness, model/hash manifests, independent replay of selected test episodes and API/UI interactions. Save initial and selected models, every training curve, complete per-scenario tables, seed metadata and report figures.

Deliver a local interactive simulator, architecture inspector, downloadable trained checkpoints, machine-readable results, detailed report, and a first-person portfolio case study with explicit negative findings and limitations.

## Sources

- MaleCNS v1.0, CC BY 4.0: https://male-cns.janelia.org/download/ — actual seed weights, source neuron identifiers and original file hashes are included.
- Google Research introduction: https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/
- EnergyPlus engineering heat-balance reference: https://energyplus.readthedocs.io/en/stable/guides/engineering-reference/2.1-basis-for-the-zone-and-air-system-integration.html
- BOPTEST KPI definitions: https://ibpsa.github.io/project1-boptest/docs-design/core_kpi.html — informs separate energy/discomfort reporting; not a claim of BOPTEST validation.
- Differentiable policy learning: https://proceedings.mlr.press/v139/mora21a.html — methodological background; this implementation is independent and simpler.
