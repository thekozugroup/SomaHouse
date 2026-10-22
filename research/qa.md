# Release checks — 22 October 2026

## Executable evidence

- 30 unit/API/artifact tests passed locally on Python 3.12.9 / PyTorch 2.10.0.
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

## Teacher study and neutral preview extension

- Plan committed at 89a8294; frozen selection at 9bfad28 before new tests.
- Three fresh seeds; 15 selected stage checkpoints and six initial checkpoints.
- 10,800 updates; 12,902,400 reward transitions plus 165,888 data-collection transitions.
- 3,332 new policy episodes / 959,616 evaluation transitions.
- All 1,700 normal episodes independently replayed exactly. Selected checkpoint hashes, exact save/reload, finite tensors and active recurrent weight changes passed.
- Two new full traces / 576 physical steps independently integrated in NumPy; all nine paired comparisons recomputed from CSV.
- Teacher-guided reward training: cost 7.45% below reward-only and 10.05% below imitation-only. Tuned-teacher superiority gate failed: mean cost 1.34% higher, interval spans zero, temperature/CO2/light non-inferiority failed despite 8.76% lower electricity.
- Grayscale colors across HTML/CSS/JS, figures, portable entry and actual model video. Room trajectories use solid/dashed/dotted lines; weight signs use dark/light gray and negative edge dashes.
- Actual study-2 representative (seed 141), fixed test scenario 7000001. All 288 video decisions match the saved trace exactly; cell, hub and port activations replayed from the same checkpoint.
- H.264 MP4: 1280×720, 24 fps, 1,728 frames, 72 seconds. Full FFmpeg decode passed. 48 simulated hours at clearly labeled 2,400× simulation speed. No online learning or biological recordings implied. Poster and comparison figures visually inspected.
- Browser: latest neural model and tuned teacher selectable; inspector exposes all 39 selected checkpoints across two studies, actual source IDs and weights; case study displays both studies, full metrics and failed success gates. Standalone docs entry loads without backend.
- Final browser check: newest model advances to Day 1 02:10; three line styles match the legend; tuned teacher operates separately; inspector shows reward seed 141 / selected update 500 and links the correct study-2 selection.
- Downloaded study-2 checkpoint fingerprint equals the saved model: ddae080670553e29ffbf491dbe0a81250831df72a21c0ed76bd37238a86d4866.
- Study-2 video played to 72 seconds / ended without browser media error. All five case-study images and portable video loaded. Browser console has no errors/warnings in the checked house tab.
- Complete ZIP integrity checked: all 69 checkpoint files, both protocols, study evidence, video and source included.
