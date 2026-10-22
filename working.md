# SomaHouse

Active goal: build and train a modular fly-derived controller for a simulated house, with rigorous comparisons and a portfolio case study. This is a separate repository from FlyFi.

## Acceptance gates

- [x] Separate repository and frozen study protocol.
- [x] Tested heat/CO2/daylight simulator and constrained actions (16 tests).
- [x] Real hub-and-spoke architecture with source provenance.
- [x] Three-seed training, conventional/rewired controls, pruning and duplication.
- [x] Frozen selection followed by held-out scenarios and robustness evaluation.
- [x] Independent artifact readback and repeatable tests.
- [x] Working interactive house and brain inspector, full report/case study.
- [x] Study-1 public commit, remote CI and unchanged FlyFi verified.

Decision: duplication must earn its complexity. Prefer a smaller useful source circuit when extra copies do not improve held-out control.

Training: direct gradients through the custom simulator; no imitation and no claim of model-free RL. See research/protocol.md for budgets and boundaries.

Training complete: 24 candidates, 5,850 updates, 3,369,600 simulated transitions. Validation-selected family: duplicate, mean cost 0.11644, versus equal-extra-training continue 0.12101. Copies improved validation in all three seeds. This is provisional evidence; no test result has been viewed yet. Primary selection is sealed in reports/selection.json. Smaller physically cut controllers were trained and retained for comparison.

User authorized a new public GitHub repository named SomaHouse. Publish only after checks and artifact verification. GitHub authenticated owner: thekozugroup. FlyFi working tree remains clean.

## Results and verification

- 3,720 evaluated policy episodes, 1,071,360 transitions; 60 normal test seeds and 16 stress seeds reused across four shifts.
- Duplication improves test cost 4.07% against equal-extra-training continue: difference -0.00510, paired scenario 95% interval [-0.00797, -0.00289]. All three validation seeds improved before selection.
- Selected circuit: eight 32-cell spokes, 16-unit hub, 32 unique source IDs, 23,833 allocated parameters. The pruned circuit physically has four 16-cell spokes and 5,337 parameters.
- Conventional balance control remains stronger on normal energy and target compliance. Learned model uses 7.7% more energy (25.32 versus 23.52 kWh/day), with 92.7% temperature and 91.0% CO2 compliance. Dense wiring beats the same-size base fly circuit. No special biological advantage established.
- Severe weather, temperature bias, CO2 dropout and capacity reduction reveal major failures. No physical deployment or hard safety guarantees.
- All 48 checkpoints read back, recurrent weights changed, save/reload actions exact; all 180 selected normal episodes replayed exactly on training hardware. Independent NumPy replay: four full traces / 1,152 steps.
- 24 local tests passed. A regression test catches first-message leakage during spoke lesions. Corrected validation lesions only; primary/stress scores, selection and trained weights preserved. Correction hash is in results.json.
- Browser tested run/pause, advancement, gathering, target change, controller switching, hub disconnection, pruned inspection, module switching and real checkpoint download. Download hash matches the stored checkpoint. PNG charts visually inspected. Browser screenshot capture is unavailable; no narrow-screen visual claim is made.
- Full detailed report (~3,400 words), first-person case study, six figures, portable docs/ website entry and complete study bundle generated.

Next gates: final committed-state tests, public repository creation/push, remote CI verification and final clean-tree check. Future research requires new held-out seeds; target constrained control and calibrated building validation.

## Study 2 extension (requested after study 1)

Teacher-guided imitation followed by reward refinement, a tuned conventional
reference, reward-only and dense controls, three new training seeds and fresh
final scenarios. Frozen in research/teacher-protocol.md before training. Original
results remain unchanged. Finish neutral colors and a telemetry-derived preview
video, then publish the extended evidence. No guaranteed winning conclusion.

Study-1 public repository and exact-state CI passed:
https://github.com/thekozugroup/SomaHouse/actions/runs/39794524213

Progress: neutral grayscale UI and chart styles implemented; telemetry preview rendered and fully decoded (H.264, 1280×720, 24 fps, 72 seconds). Browser playback reached the end without media errors. All 288 saved-model decisions replay exactly. Study 2 is training under the registered budget; final test remains unopened. Preliminary validation does not establish superiority over the tuned teacher.

Study-2 training sealed: 15 selected stage checkpoints plus six initial models,
10,800 updates, 12,902,400 reward transitions, 165,888 data-collection transitions,
1,228,800 supervised pair presentations. Three fresh seeds. Fixed primary family:
fly-teacher-reward; representative seed 141 by median validation rank. Tuned
teacher validation cost 0.09353. Commit the selection before final evaluation.
