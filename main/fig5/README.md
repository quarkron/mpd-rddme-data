# main/fig5: `fig:phase`

**Manuscript file:** phase.pdf

Polymer-crowder phase separation from a demixed start, split out of fig:polymer on 2026-10-09 so that both print at
full width. First band: a = the model on a 10 x 5 patch of lattice (`scripts/phase_setup.py`: crowder, bath, chain
with tip growth k+/k-, free monomers and nucleation k_nuc, each species' D; every leader stops at its particle's edge)
and b = chain length over time and at t = 10 s (`demix_scene3d.lengths`). c, across the width = the whole box in 3-D
beside a slice through its middle at t = 0, 1, 10 s, finite-size over point particles (was fig:polymer d).

**Re-render** (from `scripts/`, host venv, TeX Live 2026 on PATH):

```bash
python fig_phase.py --install   # -> plots/phase.{pdf,png}, copied into the manuscript
```

**Code and data live in main/fig4** (SI/figS18 imports the same modules): `demix_scene3d.snapshots`, `demix_slab_maps.panel_maps`, `demix_lengths.panel_lengths`; runs in
`main/fig4/data/demix_noneq/slab` (from `scripts/demix_slab.sh`). This script only lays the figure out.
