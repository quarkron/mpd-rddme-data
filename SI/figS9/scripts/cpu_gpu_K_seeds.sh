#!/bin/bash
# 2026-10-01: ten seeds of the single-core reference's zero-field K_eff/K_id at the two densities of the
# CPU-vs-GPU gate (n per subvolume 0.8 and 3.1), with and without exclusion, matching the GPU sweep
# (mpd_crddme/rxK_out: 32^3, c_off 200 /s, K 200 nm^3, dt 50 us, 30 000 steps, seeds 0-9), so the SI
# validation figure compares seed distributions rather than one CPU run against ten GPU runs.
OUT=/raid/racda/campaigns/lmrddme_validation/cpu_gpu_val
B=/home/racda/ftsz_tests/studies/lmrddme_validation/benchmark/cpu_drift_rdme
export OMP_NUM_THREADS=48 OMP_PLACES=cores
for n0 in 26214 101580; do for arm in excl ideal; do for s in 0 1 2 3 4 5 6 7 8 9; do
  f=$OUT/K_${arm}_${n0}_s$s
  [ -f $f.txt ] && continue
  x=""; [ $arm = ideal ] && x="--no-exclusion"
  taskset -c 112-159 $B --dims 32,32,32 --n0 $n0 --gamma 0 --K 200 --c-off 200 --dt 5e-5 --seed $s $x \
    --steps1 15000 --steps2 15000 --observe-every 25 --observe-from 15000 --profile-out $f.txt --out $f.json
  echo "$arm $n0 s$s rc=$? $(tail -1 $f.txt | grep -o 'K_eff [0-9.]*')"
done; done; done
echo CPU_GPU_K_SEEDS_COMPLETE
