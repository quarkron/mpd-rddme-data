# SI/figS13: `fig:three-kl`

**Manuscript file:** rxthree_kl.pdf

**Re-render** (from `scripts/`, host venv, TeX Live 2026 on PATH):

```bash
python rxthree_kl_fig.py   # -> plots/{pdf,png}/rxthree_kl.*; --install copies into the manuscript
```

**What it shows:** the Kullback-Leibler divergence of the subvolume occupancy distributions of A and B from the BD
reference over the relaxation of the ternary mixture. This was block C of the main-text `fig:three-relax-readdy`
(main/fig2) until 2026-10-05 and moved to the supplement with its discussion. The build prints the medians over
tau >= 1 that the SI text quotes.

**Data:** none of its own: it imports main/fig2's pooling (`main/fig2/scripts/rxthree_data.py`, `prepare()`), so the
arms, seeds and tau grid are exactly the main figure's; `data` links to main/fig2's data, -> fig2.
