#!/bin/bash
# Per-step cost of the SI-move demixing scene in its 10 s state (D_CS_both020_si_s1), single core:
# the port (both nucleation orders, 3 reps, core 45), then ReaDDy seeded from the same state
# (1 core, core 45; then 56 threads on socket 0). No GPU.
set -u
S=/raid/racda/lmrddme_polymer/demix/D_CS_both020_si_s1.npz
C=/home/racda/ftsz_tests/studies/lmrddme_validation/advantage/cpu
O=/raid/racda/campaigns/lmrddme_validation/advantage/cpu/demix64_si_state
R=/raid/racda/campaigns/lmrddme_validation/advantage/runs
/raid/racda/opt/readdy_env/bin/python $C/make_scene.py demix $O --state-from $S | head -1
for cf in 0 1; do for r in 1 2 3; do
  taskset -c 45 $C/lmcpu $O --seed $r --steps1 3 --steps2 23 --nuc-count-first $cf --out $O/cost_cf${cf}_r$r | grep -o "per_step=[^ ]*" | sed "s/^/port cf$cf r$r /"
done; done
PREFIX=demixsistate STATE=$S /home/racda/ftsz_tests/studies/lmrddme_validation/advantage/demix_cost_state.sh sc
PREFIX=demixsistate STATE=$S /home/racda/ftsz_tests/studies/lmrddme_validation/advantage/demix_cost_state.sh cpu
echo "DEMIXSI_STATE_TIMING_COMPLETE $(date)"
