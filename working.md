# SomaHouse

Active goal: build and train a modular fly-derived controller for a simulated house, with rigorous comparisons and a portfolio case study. This is a separate repository from FlyFi.

## Acceptance gates

- [x] Separate repository and frozen study protocol.
- [ ] Tested heat/CO2/daylight simulator and constrained actions.
- [ ] Real hub-and-spoke architecture with source provenance.
- [ ] Three-seed training, conventional/rewired controls, pruning and duplication.
- [ ] Frozen selection followed by held-out scenarios and robustness evaluation.
- [ ] Independent artifact readback and repeatable tests.
- [ ] Working interactive house and brain inspector, full report/case study.
- [ ] Verified commit; FlyFi remains unchanged.

Decision: duplication must earn its complexity. Prefer a smaller useful source circuit when extra copies do not improve held-out control.

Training: direct gradients through the custom simulator; no imitation and no claim of model-free RL. See research/protocol.md for budgets and boundaries.
