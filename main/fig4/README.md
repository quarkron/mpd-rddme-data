# main/fig4: `fig:polymer`

**Manuscript file:** polymer_r2.pdf

**Re-render** (from `scripts/`):

```bash
python fig_polymer_r2.py   # -> plots/polymer_r2.{pdf,png}  (~2 min)
```

**2026-10-09 split:** fig:polymer is now statics (a, was b) and dynamics (b, was c); the setup diagram and the demixing block below are drawn into fig:phase by `main/fig5/scripts/fig_phase.py`, from the modules and data here.

**Block d (2026-10-09), now fig:phase c:** the DEMIXED slab start (crowders x < L/2, monomers x >= L/2, bath uniform; `studies/polymer/gradient_growth.py --start slab` + `gradient_slab.py`, launched by `scripts/demix_slab.sh`), two-tone slices at t = 0, 1, 10 s (`demix_slab_maps.py`; t = 0 = free monomers from the .lm initial lattice), each beside a 3-D view of the whole box at the same time (`demix_scene3d.py`, which also lays out the block: <= 2600 random occupancies per species, slice plane outlined), and the chain-length panels; data in `data/demix_noneq/slab` (from /raid/racda/lmrddme_polymer/demix_noneq). The uniform start (the earlier block d) and the droplet start are SI/figS18. Experiments that led here (one-tone, four starts): `fig4d_onetone.py`, `fig4d_noneq_compare.py`, `fig4d_noneq_polymer.py`, `fig4d_slab_modes.py` -> `plots/experiments/`.

**Plotting:** `fig_polymer_r2.py`, `chain_dynamics_campaign.py`, `chain_statics_campaign.py`, `demix_lengths.py`; common: `fcc_exact`, `chain_geometry`, `fixed_chain_harness`

**Producing the data:** Statics and crowd11: `fcc_si_campaign.sh` (-> `fcc_statics_campaign.py`, `gradient_growth.py` + `gradient_hook.py`, `gradient_bench.py`) on a frozen engine snapshot, era-struct 3fa3ae8. 5 s bath transport: `fcc_si_bath5s.sh` -> `chain_bath_snapshot3fa3ae8.py` (the snapshot copy of `chain_bath.py`, which differs from the live engine). ReaDDy crowder arm: `readdy_crowd_campaign.sh` -> `readdy_chain_bath.py` / `readdy_chain.py` (RCL_, 2 s). ReaDDy bath arm RB_: launched by hand with `readdy_chain_bath.py`, k_angle from `data/fcc_si/readdy_bath/kangle_kappa4.json`. Demixing: `demix_bath_snapshots.sh` (likely `MOVES=--si-moves TAG=_si`; **exact invocation not recorded**). Run copies and logs: `data/provenance/`.

**Data origin (copied into `data/`):** /raid/racda/lmrddme_polymer/fcc_si/{statics_*.npz, transport/T_e*, crowd11/C11_*, readdy_bath/RB_*, readdy_crowd/RCL_*}, /raid/racda/lmrddme_polymer/demix/D_{CS_both020,CVS_point020}_si_s1.*
