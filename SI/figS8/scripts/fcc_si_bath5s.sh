#!/bin/bash
# fig:polymer block c, 3 nm bath arm, re-run at 5 s (2026-09-30). On the SI move set (no rigid rotation) the dilute
# tau_1 is 0.09 s, ~3x the earlier set's, and the crowded tau_1 of the 1 s runs sat on the estimator's ceiling
# (~T/7, and the 0.25 s ACF lag cap): the 1 s set (moved to transport_1s/) cannot resolve the slowdown. 5 s gives
# T/tau_1 >= ~15 at the densest point. Same scene, seeds, build and flags as fcc_si_campaign.sh, only --bench.
# Waits for the reaction-count GPU timing (advantage study) so no timing shares GPU 7 with this.
set -u
GPU=${1:-7}
OUT=/raid/racda/lmrddme_polymer/fcc_si
LM=$OUT/src_3fa3ae8; BUILD=/lm/build_container
until grep -q "PASS COMPLETE" /raid/racda/campaigns/lmrddme_validation/advantage/runs/campaign_pairs_gpu.out 2>/dev/null; do sleep 30; done
[ -d $OUT/transport_1s ] || { mv $OUT/transport $OUT/transport_1s && mkdir -p $OUT/transport; }
cp -p "$0" $OUT/fcc_si_bath5s.sh.provenance
TCOMMON="--n 11 --kappa 4 --seed-kind exact --chains 8 --nx 32 --ny 32 --nz 32 --tau-us 20 --d-bath 0.3 --d1-um2 0.3 --bench 250000 --frame-steps 25 --move-mask 27 --no-rigid-rotation"
run() {   # run <eta> <seed>
  local f=$OUT/transport/T_e$1_s$2
  [ -f $f.npz ] && return
  docker run --rm --gpus device=$GPU -v $LM:/lm -v /home/racda/ftsz_tests:/ftsz -v /raid/racda:/raid/racda -w /lm/testing/engine \
    -e PYTHONPATH=$BUILD:$BUILD/pylm:/lm/testing/engine:/ftsz/studies/common -e STRUCT_GPU=0 -e STRUCT_INTERIOR_MASK=0x1B lm2.6:latest \
    bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && python3 chain_bath.py $TCOMMON --eta $1 --seed $2 --out $f.npz" > $f.log 2>&1
  echo "c bath 5s eta $1 seed $2 rc=$?"
}
JOBS=(); for eta in 0 0.05 0.10 0.15; do for s in 1 2 3 4 5 6 7 8; do JOBS+=("$eta $s"); done; done
( for ((k = 0; k < ${#JOBS[@]}; k += 2)); do run ${JOBS[$k]}; done ) &
( for ((k = 1; k < ${#JOBS[@]}; k += 2)); do run ${JOBS[$k]}; done ) &
wait
echo "FCC_SI_BATH5S_COMPLETE $(date)"
