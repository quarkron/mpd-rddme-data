#!/bin/bash
# Lattice per-step cost of the demixing scene in its LATE state, where the chains are built.
#
# The Sep 25 re-time (demix_cost_retime.sh) differenced t_end 0.05 and 0.15 s, i.e. a box of free monomer
# with ~100 nuclei, while the ReaDDy arm it was divided by was seeded fully polymerised. This one prices
# the state the paper reports: two runs of the published scene (same seed) stopped at 8 and 10 s, one frame
# each so the recorder costs nothing in between; wall(10) - wall(8) over the 1e5 steps between them is the
# cost with ~2500 chains holding ~44k subunits. The t = 10 s npz also carries the chain-length histogram
# the ReaDDy arm is seeded with (adv_readdy_demix.py --lengths-from).
#
#   usage: demix_cost_late.sh      (GPUs 3 5 7; 6 is reserved by the owner)
set -u
O=/raid/racda/campaigns/lmrddme_validation/advantage/runs
LM=/raid/racda/worktrees/Lattice_Microbes_2.6/demix_engine
BUILD=/lm/build_container
CHEM="--n-monomers 60000 --k-nuc 0.02 --k-sever 0 --k-poly 20 --k-depoly 3 --kappa 4"
# MOVES="--si-moves" times the SI move set; PREFIX names its outputs (the Sep 28 set, earlier moves, is demixlate).
MOVES=${MOVES:-}
PREFIX=${PREFIX:-demixlate}

lat() {   # $1 gpu, $2 seed, $3 t_end
  local f=$O/${PREFIX}_lat_s$2_t$3
  [ -f $f.npz ] && return
  docker run --rm --gpus device=$1 -v $LM:/lm -v /home/racda/ftsz_tests:/ftsz -v /raid/racda:/raid/racda \
    -w /ftsz/studies/polymer -e PYTHONPATH=$BUILD:$BUILD/pylm:/lm/testing/engine:/ftsz/studies/common \
    -e STRUCT_GPU=0 lm2.6:latest bash -lc \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && python3 gradient_growth.py \
     $CHEM $MOVES --t-end $3 --frame-dt $3 --nx 64 --subhop-q 2 --crowder-mode smear \
     --phi-core 0.20 --phi-edge 0.20 --bath-eta 0.05 --seed $2 --out $f.npz" > $f.log 2>&1
  echo "lat seed $2 t $3 rc=$? $(grep -oE 'wall [0-9]+s' $f.log | head -1)"
}

if [ -n "${JOBS:-}" ]; then          # JOBS="gpu:seed:t_end ..." run in order, one at a time per GPU
  for j in $JOBS; do IFS=: read g sd te <<< "$j"; lat $g $sd $te; done
else
  lat 3 1 10 &
  lat 5 1 8 &
  ( lat 7 2 10; lat 7 2 8 ) &
  wait
fi
echo "DEMIX_COST_LATE_COMPLETE $(date)"
