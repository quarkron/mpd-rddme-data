# SI/figS9: `fig:cpu-gpu-validation`

**Manuscript file:** cpu_gpu_validation.pdf

**Re-render** (from `scripts/`):

```bash
python cpu_gpu_validation_fig.py   # -> plots/
```

**Plotting:** `cpu_gpu_validation_fig.py`

**Producing the data:** CPU K seeds: `cpu_gpu_K_seeds.sh` -> `cpu_drift_rdme` (`g++ -O3 -march=native -fopenmp -o cpu_drift_rdme cpu_drift_rdme.cpp`). GPU K: `vex_reactive.py` (same `rxK_out` sweep as S12). `gpubr_*.npz` (from `vex_reactive.py --nper 3.125 --seed 7 --gamma 0.3,0.3,0.3 --wavelength 80 --K 200 --steps 3600 --c-off {200,2e4,5e5}`) and `val_slope_*.txt` (from `cpu_drift_rdme`, run by hand; parameters in the matching JSONs and `VALIDATION.md`): **no driver script exists for these; the commands were rebuilt from the metadata.** `bench_campaign.py` shows the container invocation.

**Data origin (copied into `data/`):** studies/lmrddme_validation/benchmark/out, /raid/racda/campaigns/lmrddme_validation/cpu_gpu_val, /raid/racda/campaigns/lmrddme_validation/mpd_crddme/rxK_out
