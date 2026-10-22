# SomaHouse

A modular fly-derived circuit learns to regulate a simulated house: temperature,
fresh air and daylight-aware lighting, coordinated through a central learned hub.

**Study 1 result:** doubling the spokes improved held-out cost by **4.1%** against an
equally trained smaller circuit. The stronger conventional balance controller
still used less energy and met more comfort targets. Dense artificial wiring
also beat the same-size fly-derived base. This is a simulation study, with
negative findings included.

![Control comparisons](reports/comparison.png)

- [Full research report](reports/report.md): equations, reward, source provenance,
  confidence intervals, all training seeds, failures and limitations.
- [Portfolio case study](reports/case-study.md): first-person project narrative.
- [Portable website entry](docs/index.html): static HTML, fonts, plots, data and
  project metadata in `docs/`. Works without the simulator backend.
- [Frozen study protocol](research/protocol.md),
  [raw episode results](reports/episodes.csv),
  [selection record](reports/selection.json),
  [independent verification](reports/verification.json).

## Watch an actual test

[72-second test preview](somahouse/static/preview.mp4): house sensors on the left,
measured circuit activity on the right, timestamped actions below. Every decision
is replayed from a saved checkpoint and checked against its recorded test trace.
48 simulated hours play at **2,400× simulation speed**; no learning happens during
playback. [Video provenance](reports/preview.json). The lab and portable site embed
the video with native playback controls. All outputs use a neutral grayscale palette.

To rerender, install FFmpeg and run `uv run --no-sync python -m somahouse.preview`
(or `--study 2` for the teacher-study representative), then regenerate the report.

## Run the lab

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended. The saved models
are included; no training or dataset download is required to use the lab.

```sh
uv sync --extra test
uv run --no-sync python -m somahouse.app --port 8766
```

Open [the house lab](http://127.0.0.1:8766/),
[the trained circuit inspector](http://127.0.0.1:8766/brain.html), or
[the case study](http://127.0.0.1:8766/case-study.html).

Change temperature targets, outside weather, daylight, occupancy and sensor or
equipment faults. Disconnect the hub or a spoke. Inspect the trained models from both studies,
real source IDs, private recurrent weights and structural lineage. Compare
against conventional controls. Download trained checkpoints or the study bundle.

Page interactions change a local simulation session. **They do not retrain the
saved model.** There is no connection to physical home equipment.

On Linux, a smaller CPU-only installation can use a normal virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m somahouse.app --port 8766
```

## Study design

The three-room simulator uses explicit heat and CO₂ balances, changing synthetic
weather/occupancy/targets, daylight, constrained signed heating/cooling, vents
and lights. The control interval is ten simulated minutes.

Four functional spokes receive different sensor streams. Each spoke starts from
actual MaleCNS neuron-pair synapse counts. An artificial 16-unit hub receives
8-dimensional messages from each domain and broadcasts twice per decision.
This hub is engineered; it is not a fly cortex. There is no persistent learned
hidden state between decisions.

The selected model has eight 32-cell spokes (256 units), plus a 16-unit hub,
and 32 unique source neuron IDs reused across copies. Source IDs remain
traceable. Learned weights can change signs; no electrophysiology is modeled.

- **24 candidates / three training seeds / 5,850 updates.**
- **3,369,600 training transitions**, from randomized four-hour windows.
- Five base architectures: fly32, fly64, exact-degree rewired32, dense32, nohub32.
- Three matched descendants per seed: continue, physical pruning, duplication.
- Direct policy optimization through differentiable physics, without imitation.
  This is model-based learning, not model-free PPO.
- Validation-selected checkpoints and family frozen before opening the test set.
- 60 normal 48-hour test scenarios and four stress families of 16 scenarios.
- 3,720 evaluated policy episodes: 1,071,360 transitions across 76 unique exogenous
  seed IDs and 124 scenario/shift combinations.
- Separate comfort, CO₂, light, energy, movement, latency and cost reporting.

| Primary normal-test mean | Learned duplicated circuit | Conventional balance |
|---|---:|---:|
| Weighted cost ↓ | 0.1205 | 0.1172 |
| Electricity, kWh/day ↓ | 25.32 | 23.52 |
| Occupied temperature compliance | 92.7% | 95.3% |
| Occupied CO₂ compliance | 91.0% | 99.97% |
| Occupied light compliance | 98.0% | 100.0% |

The learned controller used **7.7% more electricity** than balance control. Its
weighted-cost difference from balance has a paired scenario interval spanning
zero. The duplication gain versus equal-extra-training continue is 4.07%, with
cost difference −0.00510, 95% paired scenario interval [−0.00797, −0.00289].
Three training seeds are limited; these intervals condition on those seeds.

Sensor faults expose major weaknesses. Both controllers fail under biased
temperature readings. Lower aggregate cost can coincide with worse CO₂
compliance. These soft targets are not safety guarantees. This is an independent
reduced-order model, not a calibrated building, real-home trial, or official
BOPTEST benchmark. No general advantage from biological wiring is established.

## Study 2: teaching before reward training

**Result:** teacher-guided reward training reduces cost **7.4% versus reward-only**
and **10.1% versus imitation alone**. It has **1.34% higher cost than the tuned
conventional teacher**, with an uncertainty interval spanning zero. It uses
**8.8% less electricity**, but meets fewer temperature and CO₂ targets. The
registered superiority test **did not pass**. Fly versus dense cost differences
also do not establish a connectome advantage.

A separate registered extension trains standalone fly-derived models by imitating
a tuned conventional teacher, then improving through simulator reward feedback.
It compares imitation-only, reward-only and dense controls across three new
training seeds. The teacher is absent at neural inference. Original study scores
remain unchanged. The final test has **100 fresh 48-hour scenarios**, plus four
fault shifts with 24 scenarios each. A superiority claim requires lower cost,
consistent seed improvement, and registered comfort/energy non-inferiority gates.

- [Full teacher-study report](reports/teacher/report.md).
- [Registered plan](research/teacher-protocol.md).
- [Every episode](reports/teacher/episodes.csv), [results and gates](reports/teacher/results.json),
  [frozen selection](reports/teacher/selection.json), [readback](reports/teacher/verification.json).

## Verify the saved experiment

```sh
uv run --no-sync python -m pytest -q
uv run --no-sync python -m somahouse.verify
uv run --no-sync python -m somahouse.teacher_verify
uv run --no-sync python -m somahouse.report
```

Verification reads all 48 initial/selected checkpoint files, checks hashes and
finite weights, confirms active recurrent-weight learning, reloads exact actions,
and replays all 180 selected-family normal test episodes. A separate float64
NumPy implementation recomputes four 48-hour trajectories: 1,152 physical steps.
The unit/API suite checks equilibrium, mass/energy integration, gradients,
constraints, graph degrees, hub isolation, surgery and live controls.
CI uses `somahouse.verify --portable` with explicit tolerances for CPU/BLAS
rounding differences. The included training-machine verification is strict.

To reproduce training, preserve the included experiment first: move `reports/`
and `checkpoints/` to an archive **outside** this repository and recreate empty
directories with those names. Keep the source seed and frozen protocol. Then:

```sh
uv run --no-sync python -m somahouse.study train
uv run --no-sync python -m somahouse.study evaluate
uv run --no-sync python -m somahouse.verify
uv run --no-sync python -m somahouse.teacher_verify
uv run --no-sync python -m somahouse.report
```

The study refuses to train over final results or silently replace a completed
evaluation. Further tuning requires a newly registered study and new final test
seeds. Exact library builds and hardware can affect training numerics.

To preview only the portable case study, serve `docs/` with any static server:

```sh
python3 -m http.server 8767 --directory docs
```

## Project boundary and licenses

SomaHouse is separate from FlyFi image compression. It shares an attributed
public connectome seed, not image datasets, image models or experiments.

Original code and report text: [MIT](LICENSE). MaleCNS-derived data and trained
checkpoints: CC BY 4.0, attributed to FlyEM / HHMI Janelia, University of Cambridge,
MRC LMB and Google Research. See [third-party notices](THIRD_PARTY.md) and
[source provenance](data/connectome/provenance.json). Font licenses are bundled.
