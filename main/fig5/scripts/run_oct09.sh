#!/usr/bin/env bash
# 2026-10-09: the Kip3 application rerun on the CURRENT engine (master build_container, 9466763) for the manuscript,
# replacing the Sep 10 runs. Same arguments as the Sep 10 set (read from each npz's meta); GPU 5. Output: out/oct09/.
#   ./run_oct09.sh [gpu]
set -u
G=${1:-5}; O=out/oct09; mkdir -p $O
r () { local name=$1; shift; [ -f $O/$name.npz ] && { echo "skip $name"; return; }
       ./lmrun.sh $G python kip3_run.py "$@" --out $O/$name.npz > $O/$name.log 2>&1; echo "$name rc=$? $(tail -1 $O/$name.log | cut -c1-100)"; }
r gate_attach_exc --mode attach  --arm excluded --lanes isolated --leduc fig3d_hi --tsim 200 --frame-dt 5
r gate_attach_pt  --mode attach  --arm point    --lanes isolated --leduc fig3d_hi --tsim 15  --frame-dt 0.5 --no-full-lane
r gate_dilute_exc --mode dilute  --arm excluded --lanes isolated  --tsim 60 --frame-dt 1
r gate_dilute_sw  --mode dilute  --arm excluded --lanes switching --tsim 60 --frame-dt 0.2
r gate_pass_exc   --mode passing --arm excluded --lanes isolated  --tsim 60 --frame-dt 1
r gate_pass_pt    --mode passing --arm point    --lanes isolated  --tsim 60 --frame-dt 1
r gate_sidestep   --mode dilute  --arm excluded --lanes switching --tsim 20 --frame-dt 0.05
r gate_sidestep_1pl --mode dilute --arm excluded --lanes switching --tsim 20 --frame-dt 0.05 --tracers-per-lane 1   # the SCORED sidestep gate: 10 tracers per lane mislink across lane changes (also gate_dilute_sw)
for s in 1 2 3; do
  r fig1c_pt_iso_1500_s$s  --mode traffic --arm point    --lanes isolated  --leduc fig1c --tsim 1500 --frame-dt 2 --seed $s
  r fig1c_exc_iso_1500_s$s --mode traffic --arm excluded --lanes isolated  --leduc fig1c --tsim 1500 --frame-dt 2 --seed $s
  r fig1c_exc_sw_1500_s$s  --mode traffic --arm excluded --lanes switching --leduc fig1c --tsim 1500 --frame-dt 2 --seed $s
done
r fig3dhi_exc_iso --mode traffic --arm excluded --lanes isolated --leduc fig3d_hi --tsim 600 --frame-dt 1
r fig3dhi_pt_iso  --mode traffic --arm point    --lanes isolated --leduc fig3d_hi --tsim 600 --frame-dt 1
echo "OCT09 DONE $(date)"
