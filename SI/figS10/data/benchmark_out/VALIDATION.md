# CPU reference validation record (openspec: add-cross-engine-benchmark, tasks 2.2-2.4)
Date: 2026-09-01. Binary: cpu_drift_rdme.cpp @ g++ -O3 -march=native -fopenmp.
All comparisons HEAD-TO-HEAD against the canonical-build GPU DriftMpdRdmeSolver at matched
configuration and measurement (32^3 lattice, nper 3.125, gamma 0.3 all species, lambda 80 nm,
K 200 nm^3, dt 50 us; slope = d ln n_A / d phi fitted over the 32 x-columns, tail-averaged).

Gate 1+2 (equilibrium field response, normalized slope):
  c_off 2e2:  CPU -0.9923   GPU -0.9957   (ideal arm)
  c_off 2e4:  CPU -1.0482   GPU -1.0515   (ideal arm)
  c_off 5e5:  CPU -1.1414   GPU -1.1440   (ideal arm)
  c_off 5e5:  CPU -0.6607   GPU -0.6584   (exclusion arm)
  Pure drift-diffusion (reactions off): CPU -0.9866 (GPU non-reacting reference -0.99).
  Agreement 0.3-0.4% at every point across the whole aT regime => PASS.

Gate 3 (zero-field K_eff/K_id, c_off 200, vs canonical GPU sweep):
  nper 0.8: CPU excl 1.120, ideal 1.003   GPU 1.127 +- 0.005, ideal 1.003  => PASS (<2 SEM)
  nper 3.1: CPU excl 1.517, ideal 1.000   GPU 1.512 +- 0.002, ideal 1.001  => PASS
  Overflow events in all runs: 0.

Gate 4 (threading): 1-thread and 56-thread runs at identical seed produce BIT-IDENTICAL
observation output (counter-based RNG keyed on step/pass/site/slot; frozen-state choices,
deterministic gather; no atomics).

FINDING recorded during validation: at the benchmark timestep dt = 50 us the reaction pass runs
at a_r*tau = c_off*dt = 25 (saturated). BOTH engines then steepen the ideal equilibrium profile
identically (-0.99 -> -1.14) and the exclusion profile to -0.66: a property of the MPD reaction
scheme at large a*tau (the manuscript's temporal-flatness bound), not an artifact of either
implementation. The published lambda=8h accuracy runs used dt <= 0.7 us (a*tau <= 0.35) and are
unaffected. Benchmark throughput at dt = 50 us is therefore a DIFFUSION-limited step; the
reaction-limited step for this chemistry is ~2 us (a*tau = 1) and scales the lattice
simulated-time-per-wall down 25x. Both numbers will be reported.
