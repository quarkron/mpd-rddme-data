# SI/figS3: `fig:wm-omega`

**Manuscript file:** wm6_omega_tau.pdf

**Re-render** (from `scripts/`):

```bash
python wm6_plot.py   # -> plots/{pdf,png}/
```

**Plotting:** `wm6_plot.py` (reads `data/rep_h{2,4}.npz`)

**Producing the data:** `driver.sh` (originally /raid/racda/tmp/wm6rep, outside the repo) runs `run_wm6_sweeps.py` in docker `4d-cell-numba:lm2.6` on 4 GPU legs (seeds 0-9). Its npz writes failed, so `parse_reps.py` rebuilt `rep_h{2,4}.npz` from the leg logs `data/g{A,B,C,D}.log` (it hard-codes a stale output path). `run_wm6_sweeps.py` needs `lm_build.py` on the path.

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/mpd_crddme/wm6_replicates (rep_h2.npz, rep_h4.npz, gA-gD.log)

**Note:** studies/FIGURES.md attributes this figure to `run_wm6_sweeps.npz` + the notebook; that is stale. The single-seed `wm6_omega_tau_sweep.npz` is not used.
