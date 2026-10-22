# SomaHouse

Status: both registered simulation studies completed, verified and published.
Public repository: https://github.com/thekozugroup/SomaHouse
Image-compression work remains isolated and unchanged in FlyFi.

## Completed

- Three-room heat/CO2/daylight simulator; constrained HVAC, ventilation and lights.
- Actual MaleCNS-derived private recurrent masks; engineered hub-and-spoke messaging.
- Study 1: 24 candidates, three seeds, 5,850 updates, 3,369,600 training transitions.
- Matched extra-training, physical pruning, duplication, rewired/dense and no-hub controls.
- Study 2: teacher imitation, reward refinement, reward-only and dense controls;
  15 selected stage checkpoints, six initial models and three fresh training seeds.
- Study-2 budget: 10,800 updates, 12,902,400 reward transitions, 165,888
  data-collection transitions and 1,228,800 supervised pair presentations.
- Original scores/checkpoints preserved. Study-2 plan committed at 89a8294;
  selected models sealed at 9bfad28 before opening fresh final tests.
- New final set: 100 normal scenarios plus four shifts of 24 paired scenarios.
  3,332 evaluated policy episodes / 959,616 evaluation transitions.
- Standalone trained-model house lab, source/weight inspector, both full reports,
  first-person case study, downloadable checkpoints and portable docs/ entry.
- Neutral white/charcoal/gray throughout. Room lines use solid/dashed/dotted styles.
- Actual held-out video: house left, brain right, action feed below. Study-2 seed
  141, scenario 7000001, 288 decisions checked exactly against the stored trace.
  48 simulated hours / 72 seconds, explicitly labeled 2,400× simulation speed.
  H.264, 1280×720, 24 fps; full decode and browser playback to the end passed.

## Findings

Study 1: duplication reduced held-out cost 4.07% against equal-extra-training
continue, paired 95% interval [-0.00797, -0.00289] on cost difference. Physical
pruning hurt. The conventional reference remained stronger on energy/comfort;
dense wiring beat the same-size fly-derived base.

Study 2: teacher-guided reward training reduced cost 7.45% against reward-only
and 10.05% against imitation-only. It did not pass the registered superiority
claim against the tuned conventional teacher: 1.34% higher mean cost, paired
interval spanning zero; 8.76% less electricity with worse temperature/CO2/light
compliance. No special biological-wiring advantage or real-home readiness shown.
Compare controllers within each study; their final scenario cohorts differ.

## Verification and publication

- 30 unit/API/artifact tests passed on the committed implementation.
- All 69 initial/selected checkpoint files included and verified.
- Original 180 selected-family episodes and all 1,700 new normal policy episodes
  replayed exactly on the training CPU; exact save/reload and learned recurrence.
- Six complete traces / 1,728 physical steps independently integrated with NumPy.
- All nine new paired comparisons independently recomputed from episode CSV.
- 190 artifact manifest files match. ZIP integrity checked with all checkpoints,
  both protocols, reports, source and video. All figure/poster pixels are grayscale.
- Browser checked newest model advancement, tuned teacher, styles, inspector,
  correct selection link, downloaded checkpoint fingerprint, portable entry,
  loaded figures and completed video. No checked house console/media errors.
- Browser screenshot capture unavailable; no narrow-screen visual proof claimed.
- Published implementation a2ea42f passed GitHub CI, including both replay jobs:
  https://github.com/thekozugroup/SomaHouse/actions/runs/39799796823
- Commits/push use thekozugroup <thekozugroup@gmail.com>; no AI coauthor tags.

## Next research

Register a new study with explicit comfort/air constraints and compare energy at
matched service levels. Improve plant identification and test a calibrated
external building model. Keep current final tests sealed from further tuning.
The portable website entry is ready for integration; the personal site has not
been modified or deployed by this project. Page interactions use fixed weights.
