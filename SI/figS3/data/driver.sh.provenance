#!/bin/bash
# 10-seed replicates, seeds split across 4 idle GPUs. Each leg writes its own npz so a partial
# sweep is still usable; the merge step tolerates missing legs.
set -u
REPO=/home/racda/ftsz_tests/Lattice_Microbes_2.6
OUTD=/lm/testing/mpd-crddme/figures_wm6/rep
mkdir -p "$REPO/testing/mpd-crddme/figures_wm6/rep"
launch () {  # $1=gpu  $2=seed-list  $3=tag
  local g=$1 seeds=$2 tag=$3
  mkdir -p /raid/racda/tmp/wm6rep/$tag
  ( docker run --rm --name wm6rep_${tag}_h2 --gpus "\"device=$g\"" \
      -v $REPO:/lm -v /raid/racda/tmp/wm6rep/$tag:/scratch \
      -e TMPDIR=/scratch -e WM6_GPU=0 -e WM6_SEEDS="$seeds" -e WM6_MODE=both \
      -e WM6_TAUS_US=40,80,160,200 -e WM6_OUT=$OUTD/h2_${tag}.npz \
      -w /lm/testing/mpd-crddme 4d-cell-numba:lm2.6 bash -lc "python -u run_wm6_sweeps.py"
    docker run --rm --name wm6rep_${tag}_h4 --gpus "\"device=$g\"" \
      -v $REPO:/lm -v /raid/racda/tmp/wm6rep/$tag:/scratch \
      -e TMPDIR=/scratch -e WM6_GPU=0 -e WM6_SEEDS="$seeds" -e WM6_MODE=tstep -e WM6_N=64 \
      -e WM6_TAUS_US=40,80,160,200 -e WM6_OUT=$OUTD/h4_${tag}.npz \
      -w /lm/testing/mpd-crddme 4d-cell-numba:lm2.6 bash -lc "python -u run_wm6_sweeps.py"
  ) > /raid/racda/tmp/wm6rep/$tag.log 2>&1 &
}
launch 6 0,1,2  gA
launch 7 3,4,5  gB
launch 3 6,7    gC
launch 5 8,9    gD
wait
echo "ALL LEGS DONE"
