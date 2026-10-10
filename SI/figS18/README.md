# SI/figS18: `fig:demix-starts`

**Manuscript file:** demix_starts.pdf

**Re-render** (from `scripts/`, host venv, TeX Live 2026 on PATH):

```bash
python demix_starts_fig.py   # -> plots/{pdf,png}/demix_starts.*; --install copies into the manuscript
```

**What it shows:** polymer-crowder demixing from two starts other than the main-text slab (fig:phase c), each drawn
as fig:phase c is (2026-10-09): the whole box in 3-D beside a two-tone slice through its middle
(`main/fig4/scripts/demix_scene3d.snapshots`), and beneath, the chain length over time and at t = 10 s
(`demix_lengths.panel_lengths`). (a) the uniform start (the old fig:polymer d), t = 0, 2, 10 s, data
`main/fig4/data/demix`; (b) the droplet start, t = 0, 1, 10 s, data `main/fig4/data/demix_noneq/droplet`. No
headlines. Placed at 0.81 textheight in SI.tex.

**Data:** none of its own; `data` -> main/fig4's data (`data/demix` for a, `data/demix_noneq/droplet` for b).
