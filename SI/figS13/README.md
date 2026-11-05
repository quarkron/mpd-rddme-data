# SI/figS13: `fig:three-eq`

**Manuscript file:** rxthree_eq.pdf

**Re-render** (from `scripts/`, host venv):

```bash
python rxthree_eq_fig.py   # -> plots/{pdf,png}/rxthree_eq.*; --install copies into $MANUSCRIPT_FIGURES
```

**What it shows:** the subvolume occupancy distributions of A and B at the end of the ternary relaxation (tau 20):
BD, the volume-excluded lattice (MPD-RDDME) and the point-particle lattice. This was block b of the main-text
`fig:three-relax-readdy` (main/fig2) until 2026-11-05, when the owner swapped it with this figure's earlier content,
the divergence of those distributions from BD over time (`fig:three-kl` until then; now Fig. 2b, drawn by
`main/fig2/scripts/fig2_block_b_kl.py`).

**Data:** none of its own: it imports main/fig2's pooling (`main/fig2/scripts/rxthree_data.py`, `prepare()`), so the
arms, seeds and tau grid are exactly the main figure's.
