#!/bin/bash
# ReaDDy reference for Fig 2 run to tau 20 (5x the published window), Stokes D_C, default dt 1.6e-7, stride 500.
# Run 2026-10-07/08 as two batches from studies/lmrddme_validation/readdy_ref: seeds 0-11 at 8 threads (12 concurrent),
# seeds 12-39 at 12 threads (14 concurrent). Output: /raid/racda/campaigns/lmrddme_validation/readdy_ref/out_three_relax_t20;
# the figure reads copies in ../data/readdy_ref/out_three_relax_t20 (RXTHREE_WINDOW=_t20).
#   usage: readdy_three_t20.sh <first seed> <last seed> <concurrency> <threads>
cd /home/racda/ftsz_tests/studies/lmrddme_validation/readdy_ref
export CONDA_PREFIX=/raid/racda/opt/readdy_env
R=/raid/racda/campaigns/lmrddme_validation/readdy_ref/out_three_relax_t20; mkdir -p $R
seq $1 $2 | xargs -P $3 -I{} bash -c "[ -f $R/rt3_s{}.npz ] && exit 0; /raid/racda/opt/readdy_env/bin/python3 readdy_three.py \
  --seed {} --threads $4 --tau-max 20 --out $R/rt3_s{}.npz > $R/rt3_s{}.log 2>&1; echo exit \$? s{} \$(date +%T)"
echo "CAMPAIGN DONE $(date)"
