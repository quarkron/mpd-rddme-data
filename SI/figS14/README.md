# SI/figS14: `fig:rx-K-main`

**Manuscript file:** rx_K_main.pdf

**Re-render** (from `scripts/`):

```bash
python rx_K_main_fig.py   # -> plots/{pdf,png}/
```

**Plotting:** `rx_K_main_fig.py`

**Producing the data:** ReaDDy: `readdy_rx.py --n0 {51,96,147,198,255,319} --seed s --out outm/rd_n{n0}_s{s}.npz`. Lattice: `vex_reactive.py --nper {0.8,1.5,2.3,3.1,4.0,5.0} --seed {0..9} [--no-exclusion] --out rxK_out/sw_{excl|ideal}_n{nper}_s{s}.npz`. **No drivers saved; the commands were rebuilt from the metadata.**

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/readdy_ref/outm, /raid/racda/campaigns/lmrddme_validation/mpd_crddme/rxK_out
