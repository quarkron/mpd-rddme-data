#!/bin/bash
# Lattice arms of the ternary relaxation figure: model (free-volume functional), oldfunc (the functional it replaced),
# point (no exclusion); N seeds each, sequential on one GPU (each run is ~10 s: 80 nominal steps at tau 4, adaptive tau
# stretches them to a few hundred). Needs the reference's tau = 0 joint histogram (readdy_ref/make_seed_hist.py).
#   usage: lattice_three_campaign.sh <seed_hist.npy> [seeds=10] [gpu=4] [arms="model oldfunc point"]
HIST=$(readlink -f "$1"); N=${2:-10}; GPU=${3:-4}; ARMS=${4:-"model oldfunc point"}
T=/home/racda/ftsz_tests/Lattice_Microbes_2.6; B=build_container
OUT=/raid/racda/campaigns/lmrddme_validation/mpd_crddme/relax3_out
for arm in $ARMS; do for s in $(seq 0 $((N-1))); do
  f=$OUT/lat3_${arm}_s${s}.npz; [ -f $f ] && { echo "skip $arm s$s"; continue; }
  docker run --rm --gpus device=$GPU --user $(id -u):$(id -g) -e HOME=/tmp -v $T:/lm -v /raid/racda:/raid/racda -w /lm/testing/engine \
    -e PYTHONPATH=/lm/$B:/lm/$B/pylm:/lm/src/lm_diagnostics lm2.6:latest bash -lc \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && python ternary_relax.py --arm $arm --seed $s --seed-hist $HIST --out $f" \
    > $OUT/lat3_${arm}_s${s}.log 2>&1
  echo "$arm s$s rc=$? $(grep -o 'frames, tau [0-9.e-]*..[0-9.]*, tau_scale [0-9.]*, struct_err [0-9]*' $OUT/lat3_${arm}_s${s}.log)"
done; done
echo "LATTICE CAMPAIGN COMPLETE $(date)"
