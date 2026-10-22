# SomaHouse

Active goal: build and train a modular fly-derived controller for a simulated house, with rigorous comparisons and a portfolio case study. This is a separate repository from FlyFi.

## Acceptance gates

- [x] Separate repository and frozen study protocol.
- [x] Tested heat/CO2/daylight simulator and constrained actions (16 tests).
- [x] Real hub-and-spoke architecture with source provenance.
- [x] Three-seed training, conventional/rewired controls, pruning and duplication.
- [ ] Frozen selection followed by held-out scenarios and robustness evaluation.
- [ ] Independent artifact readback and repeatable tests.
- [ ] Working interactive house and brain inspector, full report/case study.
- [ ] Verified commit; FlyFi remains unchanged.

Decision: duplication must earn its complexity. Prefer a smaller useful source circuit when extra copies do not improve held-out control.

Training: direct gradients through the custom simulator; no imitation and no claim of model-free RL. See research/protocol.md for budgets and boundaries.

Training complete: 24 candidates, 5,850 updates, 3,369,600 simulated transitions. Validation-selected family: duplicate, mean cost 0.11644, versus equal-extra-training continue 0.12101. Copies improved validation in all three seeds. This is provisional evidence; no test result has been viewed yet. Primary selection is sealed in reports/selection.json. Smaller physically cut controllers were trained and retained for comparison.

User authorized a new public GitHub repository named SomaHouse. Publish only after checks and artifact verification. GitHub authenticated owner: thekozugroup. FlyFi working tree remains clean.
