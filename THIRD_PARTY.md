# Third-party attribution

## Connectome source and derived checkpoints

MaleCNS v1.0 is from FlyEM (HHMI Janelia), the University of Cambridge, MRC LMB,
and Google Research. The dataset is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
[Official dataset and download page](https://male-cns.janelia.org/download/).

`data/connectome/seed.npz` is a derived 512-cell subset extracted in the separate
FlyFi project. `provenance.json` records the selection method, exact source
neuron IDs, source URLs, original download fingerprints and derived seed hash.
The extraction selects a learning-circuit annotation pool and grows a connected
subset. It is not the full fly brain.

SomaHouse uses 32/64-cell nested subsets and salience-selected 16-cell subsets.
Learned checkpoints transform the source weights and add artificial adapters,
readouts, an engineered hub and functional copies. `checkpoints/` and derived
connectome data retain the CC BY 4.0 license and this attribution. Checkpoint
lineage records pruning and duplication. Source transmitter labels are only
initialization heuristics; learned dynamics are not biological simulation.

## Fonts

DM Sans and Instrument Serif are bundled under the SIL Open Font License.
Their complete license notices accompany the fonts in
`somahouse/static/fonts/` and `docs/fonts/`.

## Methodological references

EnergyPlus zone heat balances and BOPTEST's separate reporting of energy and
thermal discomfort informed the design. This custom simulator is not a
validated EnergyPlus model or an official BOPTEST benchmark. No code from those
projects is bundled. The direct policy-gradient implementation is independent;
the PODS paper is cited for background, not an implementation claim.
