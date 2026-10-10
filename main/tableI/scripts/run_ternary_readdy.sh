#!/bin/bash
# ReaDDy arms of the ternary row. Run AFTER the sweep campaign: these are wall-clock measurements
# and a contended machine corrupts them. Pinned to the same cores the campaign used so the two
# systems' cost ratios are measured on identical hardware.
set -u
H=/home/racda/ftsz_tests/studies/lmrddme_validation/advantage
O=/raid/racda/campaigns/lmrddme_validation/advantage/runs
PY=/raid/racda/opt/readdy_env/bin/python
export CONDA_PREFIX=/raid/racda/opt/readdy_env
# The ternary scene carries no external field, matching the lattice arm.
for rep in 0 1 2; do
  for spec in "sc_db 1 40" "cpu_g 1 40" "cpu_g 56 0-55"; do
    set -- $spec; arm=$1; th=$2; cores=$3
    name=$arm; [ "$th" != "1" ] && name=${arm}${th}
    f=$O/ternary_rd_${name}_r${rep}.json
    [ -f "$f" ] && { echo "skip $name r$rep"; continue; }
    taskset -c $cores $PY $H/adv_readdy.py --scheme ternary --eta 0.14 --arm $arm \
      --threads $th --seed $rep --no-field --steps1 10 --steps2 50 --out "$f" 2>&1 \
      | grep -E "ADV_READDY|Error|Traceback"
  done
done
echo TERNARY_READDY_DONE
