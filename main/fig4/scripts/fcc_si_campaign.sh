#!/bin/bash
# fig:polymer blocks b and c re-run on the SI move set (LM-RDDME SI sec:fcc-moveset: end-bond rotation, full kink =
# flip + sideways, edge translation, face-vector collective hop; no crankshaft, no rigid rotation), on the demix-engine
# build, the only tree carrying the sideways and edge-translation move bits.
# 2026-09-30: runs on the structure-ERA HEAD (SERA 053, era-struct 3fa3ae8, which contains demix-engine 7b094e6), from a
# frozen `git archive` snapshot so the ERA runner cannot change the tree mid-campaign. The 3 nm bath arm runs 8 seeds. Same scenes and seeds as the earlier runs;
# outputs go to fcc_si/ so the earlier-move-set data (fcc/) stays intact.
#
#   b  fcc_statics_campaign.py --set len / kappa     (the chain is now recentred between frames: the non-periodic
#                                                     walls otherwise lower <R^2>, see sample_recentred)
#   c  chain_bath.py (3 nm bath, 4 eta x 4 seeds)     fcc_transport_campaign.sh's arguments + the SI mask
#      gradient_growth.py (20 nm crowders, 4 phi x 3) fcc_crowd11_campaign.sh's arguments + --si-moves, two at a time
#
#   usage: fcc_si_campaign.sh [gpu]      (default 7: GPUs 0-5 carry alfiap's jobs, GPU 6 is the owner's reserve)
set -u
GPU=${1:-7}
OUT=/raid/racda/lmrddme_polymer/fcc_si
LM=/raid/racda/lmrddme_polymer/fcc_si/src_3fa3ae8   # era-struct 3fa3ae8 (see COMMIT)
BUILD=/lm/build_container
mkdir -p $OUT/transport $OUT/crowd11
cp -p "$0" $OUT/fcc_si_campaign.sh.provenance

dock() {   # dock <workdir> <env...> -- <command>
  local wd=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
  docker run --rm --gpus device=$GPU -v $LM:/lm -v /home/racda/ftsz_tests:/ftsz -v /raid/racda:/raid/racda -w $wd \
    -e PYTHONPATH=$BUILD:$BUILD/pylm:/lm/testing/engine:/ftsz/studies/common -e STRUCT_GPU=0 "${envs[@]}" lm2.6:latest \
    bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && $*"
}

# ---- b: statics
for set in len kappa; do
  f=$OUT/statics_$set.npz
  [ -f $f ] || dock /ftsz/studies/polymer -- python3 fcc_statics_campaign.py --set $set --out $f > $OUT/statics_$set.log 2>&1
  echo "b statics $set: $(tail -1 $OUT/statics_$set.log)"
done

# ---- c: chain in the 3 nm bath
TCOMMON="--n 11 --kappa 4 --seed-kind exact --chains 8 --nx 32 --ny 32 --nz 32 --tau-us 20 --d-bath 0.3 --d1-um2 0.3 --bench 50000 --frame-steps 25 --move-mask 27 --no-rigid-rotation"
for eta in 0 0.05 0.10 0.15; do for s in 1 2 3 4 5 6 7 8; do
  f=$OUT/transport/T_e${eta}_s${s}
  [ -f $f.npz ] && continue
  dock /lm/testing/engine STRUCT_INTERIOR_MASK=0x1B -- python3 chain_bath.py $TCOMMON --eta $eta --seed $s --out $f.npz > $f.log 2>&1
  echo "c bath eta $eta seed $s rc=$?"
done; done

# ---- c: chains among 20 nm crowders, two runs at a time on the one GPU
CCOMMON="--crowder-mode smear --no-chemistry --seed-chains 8 --seed-n 11 --kappa 4 --t-end 10 --frame-dt 0.005 --nx 64 --subhop-q 2 --save-beads --si-moves"
crowd() {   # crowd <phi> <seed>
  local phi=$1 seed=$2 prof name
  if [ "$phi" = "0" ]; then prof="--n-crowders 0 --phi-core 0.05 --phi-edge 0.05"; else prof="--phi-core $phi --phi-edge $phi"; fi
  name=C11_phi${phi}_s${seed}
  [ -f $OUT/crowd11/$name.npz ] && return
  dock /ftsz/studies/polymer -- python3 gradient_growth.py $CCOMMON $prof --seed $seed --out $OUT/crowd11/$name.npz \
    > $OUT/crowd11/$name.log 2>&1
  echo "c crowd phi $phi seed $seed rc=$? $(grep -o 'done:.\{0,40\}' $OUT/crowd11/$name.log)"
}
JOBS=(); for s in 1 2 3; do for p in 0 0.05 0.10 0.15; do JOBS+=("$p $s"); done; done
( for ((k = 0; k < ${#JOBS[@]}; k += 2)); do crowd ${JOBS[$k]}; done ) &
( for ((k = 1; k < ${#JOBS[@]}; k += 2)); do crowd ${JOBS[$k]}; done ) &
wait
echo "FCC_SI_CAMPAIGN_COMPLETE $(date)"
