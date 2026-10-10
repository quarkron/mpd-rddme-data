#!/usr/bin/env bash
# 2026-10-09 (b): the Kip3 application at Leduc's 3.2 nM PER SITE (kbar_on = 3.2e-4 per step, their Fig. 1C value),
# not their Fig. 1C kbar_on N = 0.24 spread over our 1000 sites (= 2.4 nM per site). rho_s 0.91, kbar_end 0.2 as
# before. kon_N = 3.2e-4 x sites. 1800 s = Leduc's 30 min. 10 um: three arms x 3 seeds (the figure); 4, 6, 8 um:
# excluded arm x 3 seeds (steady jam vs microtubule length, Leduc Fig. 2C inset). GPU 5. Output: out/oct09b/.
set -u
G=${1:-5}; O=out/oct09b; mkdir -p $O
r () { local name=$1; shift; [ -f $O/$name.npz ] && { echo "skip $name"; return; }
       ./lmrun.sh $G python kip3_run.py "$@" --out $O/$name.npz > $O/$name.log 2>&1; echo "$name rc=$? $(tail -1 $O/$name.log | cut -c1-100)"; }
K="--rho-s 0.91 --kend-bar 0.2 --tsim 1800 --frame-dt 2"
for s in 1 2 3; do
  r c32_pt_iso_L1000_s$s  --mode traffic --arm point    --lanes isolated  --kon-N 0.32 --sites 1000 $K --seed $s
  r c32_exc_iso_L1000_s$s --mode traffic --arm excluded --lanes isolated  --kon-N 0.32 --sites 1000 $K --seed $s
  r c32_exc_sw_L1000_s$s  --mode traffic --arm excluded --lanes switching --kon-N 0.32 --sites 1000 $K --seed $s
done
for L in 400 600 800; do for s in 1 2 3; do
  r c32_exc_iso_L${L}_s$s --mode traffic --arm excluded --lanes isolated --kon-N $(python3 -c "print(round(3.2e-4*$L,4))") --sites $L $K --seed $s
done; done
echo "OCT09B DONE $(date)"
