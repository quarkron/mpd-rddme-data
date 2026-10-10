# SI/figS8: `fig:lp-match`

**Manuscript file:** lp_match.pdf (section `sec:si-chain-match`)

**Re-render** (from `scripts/`, host venv, TeX Live 2026 on PATH):

```bash
python lp_match_fig.py   # -> plots/{pdf,png}/lp_match.*; --install copies into the manuscript
```

**What it shows:** (a) the lattice and off-lattice chains; (b) lattice L_p/<b> vs kappa, closed form
(`common/fcc_exact.lp_over_b`) and the simulated lattice chain (main/fig4 `data/fcc_si/statics_kappa.npz`, the paper's
move set), one-step estimator -1/ln<cos>; (c) ReaDDy L_p/<b> vs k_theta from the converged scan, the kappa = 4 target
4.4005, the matched k_theta = 2.1105, and the k_theta = 2.3054 that fig:polymer c's reference used (L_p/<b> 4.77 +- 0.05,
+8.4 %). The build prints these numbers.

**Producing the data (2026-10-09):** `lp_scan.py` (uses the copied `readdy_chain.py`, unchanged from main/fig4):
11 beads (the fig:polymer c chain), 13 k_theta, 8 seeds, coil AND straight start each, 3.2e6 steps of 30 ns; also 32
beads (`lp_scan_n32`, the length the Sep match used). Coil and straight starts agree within one SE at every k.
Data: `data/lp_scan_n{11,32}.{npz,log}`.

**Why this exists:** the Sep 10 root find (4e5 steps, coil only) gave k_theta 2.3054, which fig:polymer c's ReaDDy arms
(readdy_crowd_campaign.sh, readdy_chain_bath.py) use; converged, that stiffness is 8 % stiff. Check runs at 2.1105 (the
dilute chain and the eta 0.15 crowder point): /raid/racda/lmrddme_polymer/fcc/readdy_kcheck (run.sh), Oct 9-10.
