# main/tableI: `tab:cost`

**Manuscript file:** (table in main.tex)

**Plotting:** **No script.** The table was assembled by hand; each entry is a median over 3 repetitions in the files under `data/`. Ternary: GPU `runs/ternary_lat_model_r*.json`, 1-core lattice `cpu_tern_s0/cost_r*.json`, ReaDDy `runs/ternary_rd_{sc_db,cpu_g56}_r*.json`. Reversible binding: `runs/summary.json`, rev row at eta 0.09, `cost` (lat_gpu, lat_cpu1, rd_sc, rd_cpu56). Demixing: GPU = (wall t10 - wall t8)/1e5 steps from `runs/demixsi_lat_s1_t{10,8}.log`; 1-core lattice `cpu_demix64_si_state/cost_cf0_r*.json`; ReaDDy `runs/demixsistate_rd_{sc,cpu}_r*.json`.

**Producing the data:** `adv_ternary.py`, `run_ternary_readdy.sh`, `adv_campaign.py` / `adv_lattice.py` / `adv_readdy.py` / `adv_cpu.cpp`, `demix_cost_late.sh` (driven by `demixsi_gpu7.sh`, running `gradient_growth.py` from ../fig4/scripts), `demixsi_state_timing.sh` -> `cpu/make_scene.py`, `cpu/lmcpu` (build with `make` in `cpu/`), `demix_cost_state.sh` -> `adv_readdy_demix.py` + `adv_demix_scene.py`.

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/advantage/runs, /raid/racda/campaigns/lmrddme_validation/advantage/cpu/{tern_s0,demix64_si_state}
