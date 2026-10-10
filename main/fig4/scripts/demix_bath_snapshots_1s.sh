#!/bin/bash
# Regenerate the polymer manuscript figure's demixing arms WITH the 3 nm bath present.
#
# WHY A RE-RUN AND NOT A RELOAD. The existing arm C run (D_C_both_s1) carries the bath but was run
# without --snap-frames, so it has no spatial snapshots at all and the figure's slice maps cannot be
# built from it. Only D_S_snap020_s1 kept snapshots, and that arm has no bath. So the bath version
# has to be run.
#
# Two arms, matching the figure's two rows and adding --bath-eta 0.05 to each:
#   CS   volume exclusion on, 20 nm bodies at phi 0.20, plus the bath
#   CVS  the point-particle control, same crowder count, exclusion off, plus the bath
# Everything else is the published configuration, including t_end 10 s and six snapshots.
# 2026-10-08 copy: SNAPS sets the snapshot count; SNAPS=11 OUT=/raid/racda/lmrddme_polymer/demix_snap1s gives
# snapshots every 1 s (t = 0..10) for the one-tone t = 0, 1, 10 experiment of block d.
set -u
OUT=${OUT:-/raid/racda/lmrddme_polymer/demix}
# THE ORIGINAL BUILD, deliberately. The demixing study was run against the ERA structure worktree,
# and the hook calls setStructureBeadBodyExclusion, which exists only there: the main tree's build
# raises AttributeError on the point-particle arm. Regenerating a published figure on the engine it
# was made with is also the right choice for provenance, and the body-hop sensor added to the main
# tree is irrelevant here, since this run measures no ceiling.
LM=${LM:-/raid/racda/worktrees/Lattice_Microbes_2.6/era_struct}
BUILD=${BUILD:-/raid/racda/ERA/struct/builds/cand2}
TEND=${TEND:-10}
SEED=${SEED:-1}
CHEM="--n-monomers 60000 --k-nuc 0.02 --k-sever 0 --k-poly 20 --k-depoly 3 --kappa 4"
# MOVES="--si-moves" runs the SI move set (full kink, end-bond rotation, edge translation, face hop; no
# crankshaft, no rigid rotation), with TAG naming the outputs apart from the earlier-move-set runs.
MOVES=${MOVES:-}
TAG=${TAG:-}
COMMON="$CHEM $MOVES --t-end $TEND --frame-dt 0.25 --nx 64 --subhop-q 2 --snap-frames ${SNAPS:-6} --bath-eta 0.05"

run () {   # run <gpu> <name> <arm flags...>
  local gpu=$1 name=$2; shift 2
  [ -f "$OUT/$name.npz" ] && { echo "skip $name"; return; }
  docker run --rm --gpus "device=$gpu" -v $LM:/lm -v /home/racda/ftsz_tests:/ftsz \
    -v /raid/racda:/raid/racda -w /ftsz/studies/polymer \
    -e PYTHONPATH=$BUILD:$BUILD/pylm:/lm/testing/engine:/ftsz/studies/common -e STRUCT_GPU=0 \
    lm2.6:latest bash -lc \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && \
     python3 gradient_growth.py $COMMON $* --seed $SEED --out $OUT/$name.npz" \
    > "$OUT/$name.log" 2>&1
  echo "$name rc=$? $(grep -o 'done:.*' $OUT/$name.log | head -1 | cut -c1-90)"
}

run "${1:-0}" D_CS_both020${TAG}_s${SEED}  --crowder-mode smear --phi-core 0.20 --phi-edge 0.20 &
run "${2:-1}" D_CVS_point020${TAG}_s${SEED} --crowder-mode point --sigma-crowd 10 --d-crowd 0.04 \
      --n-crowders 12516 --phi-core 0.20 --phi-edge 0.20 --no-vex &
wait
echo "DEMIX_BATH_SNAPSHOTS_COMPLETE $(date)"
