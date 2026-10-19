#!/bin/bash
# 2026-10-02 operator-splitting tau-ladder, fixed against randomized operator order on ONE build (rand-split, gate off =
# byte-identical to era-struct 3fa3ae8 on all Tier-0 configs). The Fig. 3 system (A+B<=>C in the sinusoidal field,
# exclusion on), adaptive tau OFF, 16 seeds, dt = 80, 40, 20, 10 ns, two windows:
#   short  tau_max 0.008, 20 frames: frames every 0.4 us = 5/10/20/40 steps, so they coincide across dt and resolve
#          the chemical relaxation (~2 us at k_off 5e5), where diffusion<->reaction splitting enters
#   long   tau_max 0.6, 60 frames: the diffusive build-up of the profile into the field, and the late K_eff
#   usage: tau_ladder_rx2.sh        GPU 7; outputs tau_ladder/<order>_randsplit/<window>/
set -u
BUILD=/raid/racda/campaigns/lmrddme_validation/tau_ladder/build_rand_split
LM=/home/racda/ftsz_tests/Lattice_Microbes_2.6
ROOT=/raid/racda/campaigns/lmrddme_validation/tau_ladder
for ORDER in fixed perm; do for WIN in short long; do
  OUT=$ROOT/${ORDER}_randsplit/$WIN; mkdir -p $OUT; cp -p "$0" $OUT/tau_ladder_rx2.sh.provenance
  if [ $WIN = short ]; then TM=0.008; FR=20; else TM=0.6; FR=60; fi
  for dt in 8e-8 4e-8 2e-8 1e-8; do for s in $(seq 0 15); do
    f=$OUT/rx_dt${dt}_s$s.npz; [ -f $f ] && continue
    docker run --rm --gpus device=7 -v $LM:/lm -v /home/racda/ftsz_tests:/ftsz -v /raid/racda:/raid/racda \
      -w /ftsz/studies/lmrddme_validation/mpd_crddme -e PYTHONPATH=$BUILD:$BUILD/pylm:/lm/testing -e LM_BUILD=$BUILD \
      -e VEX_ADAPTIVE_TAU=0 -e VEX_ORDER=$ORDER lm2.6:latest bash -lc "source /opt/conda/etc/profile.d/conda.sh && \
      conda activate lm_2.6 && python3 vex_reactive.py --nper 3.125 --sigma 3,3,3.78 --K 200 --c-off 5e5 \
      --gamma 0.3,0.3,0.3 --wavelength 80 --dt $dt --tau-max $TM --frames $FR --timeseries --seed $s --gpu 0 --out $f" \
      > ${f%.npz}.log 2>&1
    echo "$ORDER $WIN dt=$dt s=$s rc=$?"
  done; done
done; done
echo "TAU_LADDER2_COMPLETE $(date)"
