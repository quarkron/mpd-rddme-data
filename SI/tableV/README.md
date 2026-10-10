# SI/tableV: `tab:advantage`

**Manuscript file:** (table in SI.tex)

**Re-render** (from `scripts/`):

```bash
python adv_table.py   # -> plots/adv_table.tex (identical to published/adv_table.tex)
```

**Plotting:** `adv_table.py` (+ `adv_analyze.py`, `adv_schemes.py`). SI.tex carries a hand-edited caption and rounds 97.13 to 97.1.

**Producing the data:** as S9 for the rev rows; ternary rows from `adv_ternary.py` (lattice) and `run_ternary_readdy.sh` (ReaDDy).

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/advantage/runs (summary.json, rev_eta0.090_*, ternary_*)
