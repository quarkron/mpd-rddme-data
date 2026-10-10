#!/bin/bash
# ReaDDy per-step cost of the demixing scene in the STATE the paper reports: the lattice run's own 10 s
# chains and crowder snapshot (adv_readdy_demix.py --state-from), matched physics (D_M 3e5, the chain's
# 4k-mode step 1.33e-8 s, bond sqrt(2) h). Three repetitions per arm, steps 2..12 differenced.
#   usage: demix_cost_state.sh sc | cpu      (sc on core 45; cpu = 56 threads on socket 0, cores 0-55)
set -u
O=/raid/racda/campaigns/lmrddme_validation/advantage/runs
H=/home/racda/ftsz_tests/studies/lmrddme_validation/advantage
PY=/raid/racda/opt/readdy_env/bin/python
export CONDA_PREFIX=/raid/racda/opt/readdy_env
STATE=${STATE:-/raid/racda/lmrddme_polymer/demix/D_CS_both020_s1.npz}
case $1 in
  sc)  ARGS="--arm sc";  CORES=45 ;;
  cpu) ARGS="--arm cpu --threads 56"; CORES=0-55 ;;
esac
for rep in 1 2 3; do
  f=$O/${PREFIX:-demixstate}_rd_$1_r$rep.json   # PREFIX names a set apart (demixsistate = the SI-move state)
  [ -f $f ] && continue
  taskset -c $CORES $PY $H/adv_readdy_demix.py --state-from $STATE $ARGS --seed $rep \
    --steps1 2 --steps2 12 --out $f > ${f%.json}.log 2>&1
  echo "rd $1 rep $rep rc=$? $(grep -oE '"per_step_s": [0-9.]+' $f 2>/dev/null)"
done
echo "DEMIX_COST_STATE_$1_COMPLETE $(date)"
