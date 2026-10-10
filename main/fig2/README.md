# main/fig2: `fig:three-relax-readdy`

**Manuscript file:** rxthree_relax.pdf

**Re-render** (from `scripts/`):

```bash
python rxthree_relax_fig.py   # -> plots/{pdf,png}/
```

**Layout (2026-10-05):** a initial condition + starting distributions, b equilibrium distributions, c mobility of C, d lattice snapshots; every plot carries its own key. The KL panel (old c) moved to the SI as `SI/figS12` (`fig:three-kl`).

**Plotting:** `rxthree_relax_fig.py`, `rxthree_data.py` (pooling shared with S11), `rxthree_scene3d.py`; `readdy_three.py` is read at plot time (its `slab_init_mix` regenerates the tau = 0 cartoon)

**Producing the data:** ReaDDy reference: `readdy_three_campaign.sh` -> `readdy_three.py` (+ `readdy_rx.py`), 40 seeds; seed histogram: `make_seed_hist.py`. Lattice: `lattice_three_campaign.sh` -> `engine/ternary_relax.py` (+ `ribosome_blob`, `bath_species`, `friction_closure`, `bath_scene`), arms model/point, 10 seeds, container `lm2.6:latest`.

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/readdy_ref/out_three_relax (rt3_s*.npz, rt3_seed_joint.npy), /raid/racda/campaigns/lmrddme_validation/mpd_crddme/relax3_out (lat3_{model,point}_s*.npz; oldfunc arm not used)

**Note:** The lattice npz files store the absolute path of `rt3_seed_joint.npy` in `meta['seed_hist']`; the copied `rxthree_data.py` reads the local copy by basename instead.

## tau-20 window (2026-10-08; the manuscript figure since 2026-10-09)

The same scene at Stokes D_C run five times longer, to tau = 20, so panel c shows the lattice and BD mobilities meeting
(~tau 8-10) instead of stopping at tau 4.

```bash
cd scripts && python rxthree_relax_fig.py                    # tau-20 window is now the DEFAULT -> plots/{pdf,png}/
cd scripts && RXTHREE_WINDOW= RXTHREE_FIGD=../plots/t4 python rxthree_relax_fig.py   # the earlier tau-4 window
```

**Data (copied into `data/`):** `readdy_ref/out_three_relax_t20` (40 seeds, from
/raid/racda/campaigns/lmrddme_validation/readdy_ref/out_three_relax_t20: seeds 0-11 at 8 threads, 12-39 at 12 threads,
`scripts/readdy_three_t20.sh`), `mpd_crddme/relax3_out_t20` (model + point, 10 seeds each, from
/raid/racda/campaigns/lmrddme_validation/mpd_crddme/relax3_out_t20; `scripts/lattice_three_t20.sh`, which also ran the
off-Stokes D_C = 1e5 arms, kept at .../relax3_out_dc1e5_t20 and not used here). Panel c's x ticks and y range follow
the data (the default tau-4 render is byte-identical to the published figure).

Since 2026-10-09 the default window is tau 20 (RXTHREE_WINDOW unset), so SI/figS12 (KL, which imports this loader)
follows it; SI/figS13 (cross-species correlation) keeps its own copy of rxthree_data.py and still reads the tau-4 runs.
Panel d marks the drawn body's motion on the equilibrium enlargement (mark_path: start outline, recorded path
unwrapped and projected on the slice, arrow; body 0 of seed 0, 13 nm, near the median of the 64 bodies).
