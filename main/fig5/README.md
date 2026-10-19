# main/fig5: `fig:kip3-traffic`

**Manuscript file:** kip3_traffic.pdf (Kip3 motor traffic on a microtubule; the application section `sec:motors`)

**Re-render** (from `scripts/`, inside the lm2.6 container via `lmrun.sh`):

```bash
./lmrun.sh 5 python kip3_app_fig.py --pt ../data/fig1c_pt_iso_1500_s1.npz --exc ../data/fig1c_exc_iso_1500_s1.npz \
    --sw ../data/fig1c_exc_sw_1500_s1.npz --out ../plots/kip3_traffic
```

**2026-10-09 (b), now in the manuscript:** rerun at Leduc's 3.2 nM PER SITE, kbar_on = 3.2e-4 per step (their Fig. 1C
value), i.e. kon_N = 0.32 on our 1000 sites; the earlier set spread their kbar_on N = 0.24 (N = 750) over 1000 sites,
which is 2.4 nM per site. rho_s 0.91, kbar_end 0.2, 1800 s, `scripts/run_oct09b.sh` (copy; launched from
studies/lmrddme_validation/kip3_traffic on GPU 5), data `data/oct09b/`. Figure:
`kip3_app_fig.py --pt ../data/oct09b/c32_pt_iso_L1000_s1.npz --exc ../data/oct09b/c32_exc_iso_L1000_s1.npz
--sw ../data/oct09b/c32_exc_sw_L1000_s1.npz --out ../plots/kip3_traffic` (host venv, PYTHONPATH = vex-rddme/common).
Numbers: jam 4.56 +- 0.03 um (last 100 s, 3 seeds); `kip3_analyze.py slowdown --window 40 --t-min 1200 --ld 100 440
--hd 600 990`: upstream 0.90 v0 at rho 0.086, jam 0.19 v0 at rho 0.81. Length sweep (excluded, 3 seeds): 4 um 0,
6 um 0.22, 8 um 2.34, 10 um 4.56 um: slope ~1.1, critical length ~5.5 um (PFF bottleneck kend(1-kend)/kbar_on = 500
sites), against Leduc's measured slope 1 and ~2 um at 3.2 nM.

**Producing the data:** `run_oct09.sh` (gates + three 1500 s traffic arms x seeds 1-3 + the two fig3d_hi runs) ->
`kip3_run.py`, on the CURRENT master build_container (9466763), GPU 5, 2026-10-09. Seed 1 of all three arms is
bit-identical to the Sep 10 runs. Numbers in the text: `kip3_analyze.py slowdown --window 40` per seed (jam length,
upstream/jam speed, wave velocity) and `kip3_gates.py` (gates; score the sidestep gate on `gate_sidestep_1pl`: with
10 tracers per lane the linker mislinks across lane changes, which also voids `gate_dilute_sw`'s speed).

**Data origin (copied into `data/`):** /raid/racda/campaigns/lmrddme_validation/kip3_traffic/oct09 (study:
studies/lmrddme_validation/kip3_traffic, which keeps the Sep 10 runs in out/ and NOTES.md as the journal).
The synapse application that was main/fig5 until 2026-10-09 is SI/figS19. This folder was main/fig5 until 2026-10-09, when fig:phase (the polymer demixing, split out of fig:polymer) became Fig. 5.
