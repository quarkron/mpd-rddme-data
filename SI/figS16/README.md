# SI/figS16: `fig:rx-field-kl-time`

**Manuscript file:** rx_field_kl.pdf

**Re-render** (from `scripts/`, host venv, TeX Live 2026 on PATH):

```bash
python rx_field_kl_fig.py   # -> plots/{pdf,png}/rx_field_kl.*; --install copies into the manuscript
```

**What it shows:** the Kullback-Leibler divergence of the subvolume occupancy distributions of A and C from the iPRD
reference against tau, for the reversible reaction in the sinusoidal field. This was block C of the main-text
`fig:rx-field-kl` (main/fig3) until 2026-10-05 and moved to the supplement unchanged. The build prints the medians
over tau >= 1 that the SI text quotes.

**Data:** none of its own: it reads main/fig3's loaders and `data/kl3way_cache.npz` (made by `main/fig3/scripts/kl3way.py`), -> fig3.
