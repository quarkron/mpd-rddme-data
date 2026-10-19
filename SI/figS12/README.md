# SI/figS12: `fig:two-eq-readdy`

**Manuscript file:** two4_readdy.pdf

**Re-render** (from `scripts/`):

```bash
python two4_readdy_fig.py   # -> plots/{pdf,png}/
```

**Plotting:** `two4_readdy_fig.py`

**Producing the data:** ReaDDy: `readdy_two.py` (run from the directory holding `readdy_rx.py`), 16 seeds per ratio, `--na 3622 --nb {3622,1073,453,232} --sigma-b 3r`. Lattice: `vex_twospecies.py --ratio r --out vex_out2/two_r{r}.npz` (needs `lm_build.py`). **No campaign script; parameters from the npz metadata.**

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/readdy_ref/out_two, /raid/racda/campaigns/lmrddme_validation/mpd_crddme/vex_out2
