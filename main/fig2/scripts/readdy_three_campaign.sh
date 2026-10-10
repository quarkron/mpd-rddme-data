#!/bin/bash
# ReaDDy reference for the ternary relaxation figure (A 2 nm, B 4 nm, C 20 nm mobile, one C per 80 nm box).
# eta_A = eta_B = 0.07: the slab start doubles the local density and 0.10 overflows the lattice cap of 64 per voxel.
# 40 seeds, 13 concurrent x 16 threads on the 224-core host; 40-55 ms/step from the slab start -> 15-20 min per seed.
# Skips seeds already done. Outputs: readdy_ref/out_three_relax/rt3_s<seed>.{npz,log} (symlink onto /raid).
#   usage: readdy_three_campaign.sh [nseeds=40] [concurrency=13] [extra driver args...]
cd /home/racda/ftsz_tests/studies/lmrddme_validation/readdy_ref
export CONDA_PREFIX=/raid/racda/opt/readdy_env
N=${1:-40}; P=${2:-13}; shift 2 2>/dev/null; EXTRA="$*"
seq 0 $((N-1)) | xargs -P $P -I{} bash -c '
  s={}; out=out_three_relax/rt3_s${s}
  if [ -f ${out}.npz ] && grep -q "done in" ${out}.log 2>/dev/null; then echo "skip s${s}"; exit 0; fi
  /raid/racda/opt/readdy_env/bin/python3 readdy_three.py --seed $s --threads 16 '"$EXTRA"' --out ${out}.npz > ${out}.log 2>&1
  echo "exit $? s${s} $(grep -o "done in [0-9]* s ([0-9.]* ms/step)" ${out}.log)"
'
echo "CAMPAIGN COMPLETE $(date)"
