#!/bin/bash
# Fig 2 lattice arms to tau 20 (5x the figure), Stokes D_C and D_C = 1e5; master build_container, GPU 5 (idle; GPU 7 is in use), sequential.
for case in "relax3_out_t20:" "relax3_out_dc1e5_t20:--d-c 1e5"; do O=/raid/racda/campaigns/lmrddme_validation/mpd_crddme/${case%%:*}; X=${case#*:}
 for arm in model point; do for s in $(seq 0 9); do
  f=$O/lat3_${arm}_s${s}.npz; [ -f $f ] && continue
  docker run --rm --gpus device=5 --user $(id -u):$(id -g) -e HOME=/tmp -v /home/racda/ftsz_tests/Lattice_Microbes_2.6:/lm -v /raid/racda:/raid/racda -w /lm/testing/engine \
    -e PYTHONPATH=/lm/build_container:/lm/build_container/pylm:/lm/src/lm_diagnostics lm2.6:latest bash -lc \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && python ternary_relax.py --arm $arm --seed $s $X --tau-max 20 --frames 100 --seed-hist /raid/racda/campaigns/lmrddme_validation/readdy_ref/out_three_relax/rt3_seed_joint.npy --out $f" \
    > ${f%.npz}.log 2>&1
  echo "${case%%:*} $arm s$s rc=$?"
 done; done; done; echo LATTICE T20 DONE $(date)
