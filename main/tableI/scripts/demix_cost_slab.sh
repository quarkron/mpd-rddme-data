#!/bin/bash
# Table I re-time on the SLAB start (2026-10-28): the demixing state the main text and Fig. 4 show.
# The published Table I row timed the uniform start (D_CS_both020_si_s1, 2650 chains at 10 s); Sec. III C and Fig. 4
# report the slab start (demix_noneq/slab, 2198 chains at 10 s). Same methods as before:
#   GPU     demix_cost_late.sh: two runs of the slab scene (seed 1) stopped at 8 and 10 s, one frame each;
#           wall(10) - wall(8) over the 1e5 steps between them. Same engine/build as the published row
#           (demix_engine 7b094e6, /lm/build_container), SI moves, --start slab. GPUs 0 and 1, host threads pinned.
#   1 core  demixsi_state_timing.sh: the CPU port seeded from the slab run's 10 s state, both nucleation orders,
#           3 reps on core 45; then ReaDDy from the same state, 1 core (45) and 56 threads (cores 0-55).
#   control the uniform state re-timed under the same machine load (port cf0, ReaDDy 56 threads): other users hold
#           ~8 of socket 0's cores today, and the control says whether that moves the numbers.
set -u
LM=/raid/racda/worktrees/Lattice_Microbes_2.6/demix_engine
BUILD=/lm/build_container
R=/raid/racda/campaigns/lmrddme_validation/advantage/runs
C=/home/racda/ftsz_tests/studies/lmrddme_validation/advantage/cpu
CO=/raid/racda/campaigns/lmrddme_validation/advantage/cpu
SLAB=/raid/racda/lmrddme_polymer/demix_noneq/slab/D_CS_both020_si_s1.npz
UNI=/raid/racda/lmrddme_polymer/demix/D_CS_both020_si_s1.npz
CHEM="--n-monomers 60000 --k-nuc 0.02 --k-sever 0 --k-poly 20 --k-depoly 3 --kappa 4"

lat() {   # $1 gpu, $2 t_end, $3 host cpus
  local f=$R/demixsislab_lat_s1_t$2
  [ -f $f.npz ] && return
  docker run --rm --gpus device=$1 --cpuset-cpus $3 -v $LM:/lm -v /home/racda/ftsz_tests:/ftsz -v /raid/racda:/raid/racda \
    -w /ftsz/studies/polymer -e PYTHONPATH=$BUILD:$BUILD/pylm:/lm/testing/engine:/ftsz/studies/common \
    -e STRUCT_GPU=0 lm2.6:latest bash -lc \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && python3 gradient_growth.py \
     $CHEM --si-moves --start slab --t-end $2 --frame-dt $2 --nx 64 --subhop-q 2 --crowder-mode smear \
     --phi-core 0.20 --phi-edge 0.20 --bath-eta 0.05 --seed 1 --out $f.npz" > $f.log 2>&1
  echo "lat slab t $2 rc=$? $(grep -oE 'wall [0-9]+s' $f.log | head -1) $(grep -oE 'chains=[0-9]+, bound=[0-9]+' $f.log | tail -1)"
}

cpu_state() {   # $1 state npz, $2 out dir, $3 cf list, $4 ReaDDy prefix, $5 ReaDDy arms
  mkdir -p $2
  /raid/racda/opt/readdy_env/bin/python $C/make_scene.py demix $2 --state-from $1 | head -1
  for cf in $3; do for r in 1 2 3; do
    [ -f $2/cost_cf${cf}_r$r.json ] && continue
    taskset -c 45 $C/lmcpu $2 --seed $r --steps1 3 --steps2 23 --nuc-count-first $cf --out $2/cost_cf${cf}_r$r \
      | grep -o "per_step=[^ ]*" | sed "s|^|port $(basename $2) cf$cf r$r |"
  done; done
  for arm in $5; do PREFIX=$4 STATE=$1 /home/racda/ftsz_tests/studies/lmrddme_validation/advantage/demix_cost_state.sh $arm; done
}

echo "DEMIX_COST_SLAB_START $(date)"
lat 0 10 80-83 &
lat 1 8 84-87 &
cpu_state $SLAB $CO/demix64_si_slab_state "0 1" demixsislab "sc cpu"
cpu_state $UNI  $CO/demix64_si_state_ctl  "0"   demixsictl  "cpu"
wait
echo "DEMIX_COST_SLAB_COMPLETE $(date)"
