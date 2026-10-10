# main/fig3: `fig:rx-field-kl`

**Manuscript file:** rx_field.pdf

**Re-render** (from `scripts/`):

```bash
python rx_field_fig.py   # -> plots/fig_rx_field.{pdf,png}; --install copies into the manuscript
```

**Plotting:** `rx_field_fig.py`, `rx_field_data.py`, `rx_field_scheme.py` (`rx_field_voxels.py` drew the old 3D block D, kept for provenance)

**Layout (2026-10-05):** a setup, b steady profiles, c two periods of the lattice drawn flat at tau = 0, 0.053, 15: the reactant occupancy relative to its instantaneous mean beside the free-energy landscape U relative to its mean (`rx_field_data.plane_landscape`: U = sum_s n_s phi_s + F_ex, F_ex the engine's BMCSL `bFex_wb` with hard-core prefactor 1, from PLANE-MEAN occupancies, n_B = n_A by symmetry). Flat 1 x 16 strips of plane means are the only honest 2D view: an x-y slice averaged over z alone keeps ~12 % shot noise against the 17 % response. The KL divergence in time (old block C) moved to the SI: `SI/figS16` (`fig:rx-field-kl-time`), which reads this folder's data and imports `ARMS` from `rx_field_fig.py`.

**Producing the data:** Lattice: `vex_reactive.py --nper 3.125 --sigma 3,3,3.78 --K 200 --c-off 5e5 --gamma 0.3,0.3,0.3 --wavelength 80 --phase 0 --timeseries --seed S [--no-exclusion]` in windows veryearly/early/late (needs `lm_build.py` on the path). iPRD reference: `readdy_field.py --seed S` (seeds 0-47; early `--tau-max 0.12 --stride 25`, very early `--tau-max 0.02 --stride 5`), needs the compiled `sinpot/` extension. **No launcher or log survives; these commands were rebuilt from the npz metadata.** See `NOTES.md`.

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/mpd_crddme/rxl80c_out (12), /raid/racda/campaigns/lmrddme_validation/readdy_ref/out_field (144), studies/lmrddme_validation/kl3way_cache.npz
