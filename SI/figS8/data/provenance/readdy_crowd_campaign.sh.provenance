#!/bin/bash
# The off-lattice reference for the 20 nm arm: the same 11-mer among explicit 20 nm crowder spheres, so the exact
# bead-body hard core (engine SERA 025) has a Brownian-dynamics check of both the size and the mobility. The 3 nm arm
# already had one; this is its counterpart, and it is the CHEAPER scene in particle count -- 98 crowders at eta 0.15
# against 43 000 bath spheres -- because a 20 nm sphere displaces 300 times the volume of a 3 nm one.
# Matched to the lattice: 11 beads, bond sqrt(2) voxels, k_angle 2.3054 (the root find for kappa 4), bead D 0.3 um^2/s,
# crowder D 0.04 um^2/s (--d-ratio 0.1333), box 14 voxels, the same soft repulsions the bath runs use.
#   usage: [ETAS=..] [SEEDS=..] [TEND=0.3] [THREADS=12] [JOBS=6] [OUTPREFIX=RC] readdy_crowd_campaign.sh
# TEND matters for the relaxation time, not for diffusion: D_cm comes from short lags and is settled in 0.3 s, while
# tau_1 needs the run to cover many relaxation times. A half-run test on the 3 nm bath runs (0.3 s, 4 seeds) moved the
# fitted tau_1 by up to a factor of two, so those crowded points are not converged; OUTPREFIX=RCL TEND=2.0 is the wave
# that resolves it for the crowders, which cost an hour a run against the bath's twenty-three.
# The dry chain is not re-run: eta 0 has no depletant, so fcc/readdy_bath/RB_e0_s*.npz already is it.
OUT=/raid/racda/lmrddme_polymer/fcc/readdy_crowd
PY=/raid/racda/opt/readdy_env/bin/python3
HERE=$(cd "$(dirname "$0")" && pwd)
ETAS=(${ETAS:-0.05 0.10 0.15}); SEEDS=(${SEEDS:-1 2 3}); TEND=${TEND:-0.3}; THREADS=${THREADS:-12}; JOBS=${JOBS:-6}
mkdir -p $OUT; cp -p "$0" $OUT/readdy_crowd_campaign.sh.provenance 2>/dev/null
cd $HERE/readdy_ref
n=0
for eta in "${ETAS[@]}"; do for s in "${SEEDS[@]}"; do
  name=${OUTPREFIX:-RC}_e${eta}_s${s}
  [ -f $OUT/$name.npz ] && { echo "$name exists, skipping"; continue; }
  while [ $(jobs -rp | wc -l) -ge $JOBS ]; do wait -n; done
  ( $PY readdy_chain_bath.py --eta $eta --bath-sigma 2.0 --d-ratio 0.1333 --k-angle 2.3054 --box 14 \
      --t-end $TEND --threads $THREADS --seed $s --out $OUT/$name.npz > $OUT/$name.log 2>&1
    echo "rc=$?" > $OUT/$name.DONE ) &
  n=$((n+1)); echo "launched $name"
done; done
wait
echo "READDY_CROWD_DONE ($n runs)"
