# Single-core CPU port of the MPD-RDDME update rule, for the per-step cost table

Purpose: the "MPD-RDDME, 1 core" column of the main-text cost table (`tab:cost`) for the two
systems the single-file reference `../adv_cpu.cpp` cannot run: the ternary mixture (particles
larger than a subvolume) and the polymer demixing (bodies plus growing bonded chains). The
reversible-binding entry stays on `adv_cpu.cpp`, which is left untouched.

Engine ported from: `/raid/racda/worktrees/Lattice_Microbes_2.6/demix_engine` (branch
`demix-engine`, HEAD 7b094e6), the build the GPU column of the table was timed on.

## The cost contract

The number must be "the same update rule on one core". So:

1. **Every operator the GPU runs in the timed configuration runs here**, including work whose
   result is exactly zero in these scenes. The GPU evaluates it, so the CPU pays for it too:
   - the bead-body hard-core scans (per growth candidate, per destination, per nucleation voxel)
   - the friction shells (98 voxels per body attempt, 26 per moved subunit)
   - the formation-swap term
   - the chain move energies whose masked VEX reads return 0
2. **Work the GPU itself skips is skipped here.** For example, the body's smear-interaction
   energy returns 0 before any evaluation when the static baseline is empty
   (`struct_warp_hop_dev.cuh:173`), and the per-species drift memo is off by default, so every
   particle is evaluated.
3. **GPU mechanics are dropped.** These have no single-core counterpart: atomics and CAS
   claims, the 8-parity checkerboard (bodies are swept sequentially in a random order each
   step), the per-slot step stamp, host-device syncs, the label-propagation topology relabel
   (chains are kept as explicit voxel paths), diagnostic stat atomics and the word-bound
   verification scan.
4. **Oddities that ran on the GPU are reproduced, not fixed:**
   - the LINEAR hop branch in the demixing scene
   - a periodic bath beside a non-periodic structure lattice with clipped smear
   - the body centre voxel counted twice in the demixing bath densities (reciprocal fill plus smear)
   - + tip shedding only at n >= 4
   - Q = 8 (ternary) and Q = 2 (demixing)
   - per-species q from the diffusion table, never the Stokes rule
5. **Arithmetic precision follows the GPU hot path:** White Bear in fp32 on the bath hop,
   `bernoulliFast`, and the smear field in fp64.

## Divergences from the GPU (all statistical, none change the work per step)

- Random numbers come from one sequential xoshiro256+ stream, not a counter hash per draw.
  Trajectories are therefore not bit-comparable; validation is statistical.
- The bath overflow (a site above 64 particles) keeps the particle at its source. The GPU
  redistributes it host-side, and neither path runs in a healthy scene.
- The one-body field reads of the structure moves (`computeSpeciesPotential`, zero here) are
  omitted: two float loads per moved subunit, below the timing resolution.
- A chain tip's incoming bond is read from the explicit path, not by the device's 26-neighbour
  scan (`structLinearIncomingNeighbor`), which exists only because the device keeps no path.

## Layout

Every file is at most 500 lines (`make linecheck`, run by the default target).

| file | content |
|---|---|
| `rng.h` | xoshiro256+ stream |
| `wb.h` | White Bear in fp32 (bath hop) and fp64 (structure), the free-volume form, Bernoulli |
| `scene.h`, `scene_io.cpp` | scene description, and the loader for the directory `make_scene.py` writes |
| `sim.h`, `lattice.cpp` | state, particle bookkeeping, the step in the device's operator order, `LMCPU_PROFILE` |
| `bath_hop.cpp` | the three axis passes: EXACT and LINEAR branches, fill and smear fold-in, free volume |
| `structure.h`, `structure.cpp` | overlay (L1/L2/L3) and L26 encoding, seeding of bodies and chains |
| `smear.cpp` | CIC stencil deposit, full re-stamp and incremental update |
| `body.cpp` | the body's sub-voxel step, both hard cores, the collisional friction shell |
| `chain_common.h/.cpp` | bend, stretch, octahedral group, chain friction, the VEX and formation-swap terms |
| `chain_moves.cpp` | the SI move set: face-vector collective hop, end-bond rotation, full kink, edge translation |
| `chain_kinetics.cpp` | dissolution, tip retraction and shedding, tip growth, order-1 nucleation |
| `main.cpp` | driver: seeding, timing window, validation observables |
| `make_scene.py` | writes a scene directory: `ternary`, or `demix` (optionally from a lattice npz) |
| `gpu_ternary_ref.py` | the GPU reference run for the ternary validation |
| `validate_demix32.sh`, `compare_demix32.py` | the demixing kinetics validation against the GPU at 32^3 |

## One ordering choice, reported both ways

The device runs nucleation's bead-body core scan (7^3 voxels) at EVERY unstructured voxel before
it counts the monomers there. On a GPU the 262 144 scans run concurrently. On one core, counting
first gives identical outcomes, because a voxel without monomers is skipped either way, and
avoids ~94 % of the scans. `--nuc-count-first 1` selects that order. The default is the device's
order, the conservative number for the lattice. In the 64^3 state it moves the step from 0.31 s to
0.17 s, of which nucleation was 51 % (`LMCPU_PROFILE=1`).

## Validation before any timing is quoted

- ternary: per-species occupancy mean and Fano factor, and body MSD, against the GPU in the
  same scene (`gpu_ternary_ref.py`).
- demixing: chain statics against the phantom-chain closed form and the self-avoiding-walk
  reference; kinetics (chain count, bound subunits, length histogram) against a short GPU run
  at 32^3.
