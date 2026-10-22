# Release checks — 22 October 2026

## Executable evidence

- 24 unit/API/artifact tests passed locally on Python 3.12.9 / PyTorch 2.10.0.
- 48 initial and selected checkpoints read and fingerprinted.
- 24 selected checkpoints save/reload with bit-identical actions.
- Changed active recurrent weights confirmed for every trained candidate.
- 180 selected-family normal test episodes replay with exact metrics on the training CPU.
- Four complete recorded traces replayed using independent float64 NumPy physics: 1,152 steps.
- Static case-study local references and all artifact manifest hashes validated.
- Stored final scores/selection were preserved when first-message spoke lesions were corrected; regression test covers that bug.

## Real browser interaction

- House starts from the saved representative duplicated controller.
- Two-hour advancement updates time, sensors, power, energy and deficit measurements.
- Run/pause advances and stops the simulated clock.
- Gathering, setpoint adjustment, controller switch and hub disconnect work.
- Chart renders three room trajectories; no horizontal overflow at observed desktop width.
- Pruned model inspector shows 16 source cells, four spokes and 127 edges for the inspected seed.
- Selecting the air spoke changes weights, sensor description and module state.
- Actual checkpoint download hash matches checkpoints/cut-s19.pt:
  `13d7edf17477e853b20d88cd3e42e201da80a7f03e05645b3445a4166315363e`.
- Case-study copy, tables, four embedded plots and download links render in the browser.
- Comparison and capacity PNG plots inspected visually as local image artifacts.

Browser screenshot capture is unavailable in the current browser surface. Desktop DOM and interaction evidence is recorded above; no screenshot or narrow-screen visual verification is claimed. Responsive CSS and reduced-motion/focus handling are implemented.

## Scope

No home equipment connected. No real building or weather calibration. No tuning after opening the test set. FlyFi's repository and server were left unchanged. Public release is authorized by the user; author and committer are thekozugroup <thekozugroup@gmail.com>, with no coauthor tags.
