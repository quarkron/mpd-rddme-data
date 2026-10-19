# SI/figS11: `fig:advantage`

**Manuscript file:** adv_metric.pdf

**Re-render** (from `scripts/`):

```bash
python adv_fig.py   # -> plots/adv_metric.{pdf,png}
```

**Plotting:** `adv_fig.py`, `adv_schemes.py`; reads only `data/runs/summary.json` (the pr0-pr6 rows at eta 0.09)

**Producing the data:** `adv_campaign.py` (GPU arm via `adv_lattice.py` in `lm2.6:latest`; CPU arm `adv_cpu`, built from `adv_cpu.cpp`; ReaDDy arm `adv_readdy.py` + `sinpot/`). The pr sweep was launched by `data/runs/pairs_gpu_after_polymer.sh`; the CPU half is logged in `campaign_pairs_cpu.out`. Summary: `adv_analyze.py`.

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/advantage/runs (summary.json + 72 pr*_eta0.090_*.json)

**Note:** Re-running `adv_analyze.py` here rewrites `data/runs/summary.json` from the pr* files only, which is enough for this figure but not for Table V.
