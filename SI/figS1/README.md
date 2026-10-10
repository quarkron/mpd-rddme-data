# SI/figS1: `fig:wm-convergence`

**Manuscript file:** wm1_l2_convergence.pdf

**Re-render** (from `scripts/`):

```bash
python wm_sim_figs.py   # -> plots/{pdf,png}/ (also writes wm4_* for S4)
```

**Plotting:** `wm_sim_figs.py`

**Producing the data:** `wm_sim_campaign.py [partA|replicas|all]`, built against the occupancy-16 build (`data/wm_sim/build_occ16/`, commit 9466763); physics from `harmonic_trap.ipynb`. The log of the run that produced the data is `data/wm_sim/campaign_occ16.log`.

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/mpd_crddme/wm_sim (partA.npz, rep_s{0,1,2}.h5, logs)
