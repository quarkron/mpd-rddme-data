# SI/figS17: `fig:tau-ladder`

**Manuscript file:** tau_ladder.pdf (SI subsection `sec:si-splitting`, "Timestep Error of the Operator Splitting")

**Re-render** (from `scripts/`, lm2.6 container or the host venv, TeX Live on PATH):

```bash
python tau_ladder_fig.py   # -> plots/{pdf,png}/tau_ladder.*   (--install also copies into the manuscript)
```

**What it shows.** The timestep error of the Lie-Trotter step, fixed against randomized operator order, on two systems:
(a) reversible binding in the sinusoidal field (the Fig. 3 system) early in its chemical relaxation; (b) the bodies' mean
squared displacement in the ternary relaxation of Fig. 2 (bath + 64 bodies of 20 nm). The bath's stripe amplitude was a
third panel until 2026-10-05; the owner kept the body MSD alone. The figure and the SI show the fixed and the random order. Result: a
first-order error (2.3 % at 0.08 us, 0.4 % at 0.01 us in (a)), independent of the order. Both axes are in us; system 1 runs at
0.01-0.08 us because k_off = 5e5 /s sets the reaction's step (k_off dt <= 0.04).

The ternary ladder also ran the bodies' smear field re-stamped after their moves (ternary/smear, ternary/permsmear):
no effect either. Those runs are kept in data/ but are not in the figure or the SI text (owner, 2026-10-05).

**Engine.** Branch `rand-split` of Lattice Microbes 2.6 (worktree `/raid/racda/worktrees/Lattice_Microbes_2.6/rand_split`,
commit `4c8a2e2` on era-struct `3fa3ae8`; build `/raid/racda/campaigns/lmrddme_validation/tau_ladder/build_rand_split`):
`setStepOrderRandomize` and `setStructureSmearAfterMoves` (`scripts/StepOrder.cu`, copied), both default off. With both off
the ERA structure Tier-0 suite is 11/11 IDENTICAL to 3fa3ae8 (`/raid/racda/ERA/struct/pilot/rand_split_tier0`).

**Producing the data** (copies of the drivers in `scripts/`, verbatim):
- (a) `tau_ladder_rx2.sh` -> `vex_reactive.py` (`VEX_ORDER=perm` selects the random order, `VEX_ADAPTIVE_TAU=0`), 16 seeds
  x dt 80/40/20/10 ns x {short tau 0.008, long tau 0.6} x {fixed, perm}; `tau_ladder_analyze.py` prints the frame-by-frame
  comparison.
- (b, c) `tau_ladder_ternary.sh` -> engine `testing/engine/ternary_relax.py` of the rand-split worktree (copied; new flags
  `--tau-us --fixed-tau --order perm --smear-after`, defaults reproduce Fig. 2), 16 seeds x tau 50/25/12.5/6.25 us x 4 schemes.

**Data origin (copied into `data/`):** exactly the files the figure reads, with their run logs and the run script's
provenance copy.
- `data/rx/{fixed,perm}/short/rx_dt{8e-8,4e-8,2e-8,1e-8}_s{0..15}.npz` (+ `.log`): copies of
  `/raid/racda/campaigns/lmrddme_validation/tau_ladder/{fixed,perm}_randsplit/short/` (64 runs per order, panel a).
- `data/ternary/{fixed,perm}/t3_tau{50,25,12.5,6.25}_s{0..15}.npz` (+ `.log`) and `data/ternary/run.out`: copies of
  `/raid/racda/campaigns/lmrddme_validation/tau_ladder/ternary/{fixed,perm}/` (64 runs per order, panel b).

**Raw data (`raw/` -> /raid/racda/campaigns/lmrddme_validation/tau_ladder):** everything the two ladders wrote.
- `{fixed,perm}_randsplit/{short,long}/`: the binding ladder, both windows (the long window, tau 0.6, gives the late
  K_eff/K the SI quotes; `tau_ladder_analyze.py` reads it from here).
- `fixed/`: the first fixed-order binding ladder on the main build_container (9466763), a cross-build check.
- `ternary/{fixed,perm,smear,permsmear}/` + `run.out`: the full ternary ladder, including the smear-refresh-after-moves
  schemes, which the figure and the SI do not show; `ternary_smoke/`: the two smoke runs.
- `build_rand_split/`: the engine build every run used; `identity/`: the run-to-run determinism probe of the binding scene.

**Known gaps.** The body MSD (c) is 3 % noisy at 16 seeds; a trend below that is not resolved. The structure chemistry
(growth, nucleation) and chain moves are not in either system; the polymer scenes were not laddered.
