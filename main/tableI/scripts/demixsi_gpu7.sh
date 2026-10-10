#!/bin/bash
# GPU 7 queue, Sep 28 (GPUs 0-5 carry alfiap's jobs; GPU 6 reserved by the owner).
#  1. let the SI timing pair (t10 then t8, seed 1) finish
#  2. rerun t10: it shared GPU 7 for ~1.5 min with a stray validation job (script bug, fixed), so its wall is suspect;
#     the suspect files are kept as *_shared.*
#  3. the two 10 s production arms, SI moves, demix-engine build
#  4. the GPU side of the 32^3 kinetics validation, SI moves, 12 seeds
set -u
R=/raid/racda/campaigns/lmrddme_validation/advantage/runs
until grep -q DEMIX_COST_LATE_COMPLETE $R/demixsi_cost.out 2>/dev/null; do sleep 60; done
for f in $R/demixsi_lat_s1_t10.*; do mv "$f" "${f/_t10./_t10_shared.}"; done
MOVES="--si-moves" PREFIX=demixsi JOBS="7:1:10" /home/racda/ftsz_tests/studies/lmrddme_validation/advantage/demix_cost_late.sh
echo "STEP2_T10_RERUN_DONE $(date)"
LM=/raid/racda/worktrees/Lattice_Microbes_2.6/demix_engine BUILD=/lm/build_container MOVES="--si-moves" TAG=_si \
  /home/racda/ftsz_tests/studies/polymer/demix_bath_snapshots.sh 7 7
echo "STEP3_PRODUCTION_DONE $(date)"
D=/raid/racda/campaigns/lmrddme_validation/advantage/cpu/demix_val_si MOVES="--si-moves" GPUS="7" \
  SEEDS_GPU="1 2 3 4 5 6 7 8 9 10 11 12" SEEDS_CPU="" \
  /home/racda/ftsz_tests/studies/lmrddme_validation/advantage/cpu/validate_demix32.sh
echo "DEMIXSI_GPU7_COMPLETE $(date)"
