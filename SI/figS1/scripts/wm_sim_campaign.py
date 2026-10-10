"""Re-run the static harmonic-well simulations behind SI Figs. S1 (wm1_l2_convergence) and S4
(wm4_intravoxel_shape_h8), on the current DriftMpdRdmeSolver, and SAVE what the figures read.

The originals (harmonic_trap.ipynb, May 2026, build_v1) kept Part A's marginals in notebook memory
and wrote the replicate frames to /raid/racda/mpd-crddme-replicates, which is now empty, so neither
figure could be redrawn. This script is the notebook's cells 1, 3, 4 and 5 unchanged in physics:

  part A    one run per N in N_SWEEP (seed 0): the time-averaged 1-D x-marginal -> partA.npz (S1)
  replicas  seeds 0-2 at N = 512: per-frame 3-D occupancy, gzip HDF5 -> rep_s<seed>.h5 (S4)

Particle counts use PPS = 16, the compiled site width of build_v1, so every run places exactly the
particles the original did; the current build's site width only bounds the occupancy.

usage (lm2.6 container, build on PYTHONPATH):  python wm_sim_campaign.py [partA|replicas|all]
"""
import os, sys, tempfile, time
import numpy as np
import h5py
import lm
from jLM.RDME import Sim as RDMESim

OUT = os.environ.get('WM_OUT', '/raid/racda/campaigns/lmrddme_validation/mpd_crddme/wm_sim')

# ---- physics, as cell 1 (box units; 1 unit = UNIT meters) ------------------------------------
L      = 256.0
UNIT   = 1e-8
D      = 1e-12
K      = 0.01
SIGMA  = 1.0 / np.sqrt(K)
C_STAB = 0.10
PPS    = 16                      # build_v1's site width: sets n_part exactly as the original runs did
PPS_BUILD = lm.getCompiledLatticeMaxOccupancy()
N_SWEEP = [32, 64, 128, 256, 512]
N_REP, N_REPLICATES = 512, 3


def build_static_field(N):
    """phi_i = (K/2) * sum_a ((idx_a - N/2) * h)^2, in box units, C-flattened float32."""
    h = L / N
    ax = ((np.arange(N) - N / 2.0) * h).astype(np.float64)
    phi = 0.5 * K * (ax[:, None, None]**2 + ax[None, :, None]**2 + ax[None, None, :]**2)
    return np.ascontiguousarray(phi.ravel(order='C'), dtype=np.float32), h


def seed_gaussian(sim, P, N, n_part, rng):
    """Place n_part particles drawn from N(center, sigma_voxels) independently per axis."""
    h = L / N
    sigv = SIGMA / h
    c = N / 2.0
    pos = np.round(c + sigv * rng.standard_normal((n_part, 3))).astype(np.int64)
    pos = np.clip(pos, 0, N - 1)
    vox, cnt = np.unique(pos, axis=0, return_counts=True)
    cnt = np.minimum(cnt, PPS)
    for (x, y, z), n in zip(vox, cnt):
        sim.placeNumber(P, int(x), int(y), int(z), int(n))
    return sigv


def run(N, workdir, seed, run_steps=3000, n_frames=30, occ_frac=0.25, max_part=300_000):
    """Cell 3's run_static / cell 5's run_static_to_h5: returns (frames uint8 (F,N,N,N), times, n_part)."""
    h = L / N
    sigv = SIGMA / h
    dt = C_STAB * (h * UNIT)**2 / D
    n_part = int(np.clip(occ_frac * PPS * (2 * np.pi)**1.5 * sigv**3, 200, max_part))
    rng = np.random.default_rng(seed)
    fname = os.path.join(workdir, f'fpA_s{seed}_N{N}.lm')
    sim = RDMESim(name=f'fpA_s{seed}_N{N}', filename=fname, dimensions=[N, N, N],
                  latticeSpacing=h * UNIT, regionName='cyt', latticeType='Byte', dt=dt)
    P = sim.species('P'); sim.diffusionConst('D', D)
    sim.transitionRate(None, None, None, sim.diffusionZero)
    sim.transitionRate(P, sim.region('cyt'), sim.region('cyt'), sim.dc.D)
    seed_gaussian(sim, P, N, n_part, rng)
    sim.simulationTime = run_steps * dt
    sim.latticeWriteInterval = max(1, run_steps // n_frames)
    sim.speciesWriteInterval = sim.latticeWriteInterval
    psi0, _ = build_static_field(N)
    sim.finalize()
    solver = lm.DriftMpdRdmeSolver()
    solver.setPotentialBasis(psi0, np.zeros_like(psi0))
    solver.setSpeciesCoupling(0, 1.0, 0.0)
    f = sim.run(solver=solver, cudaDevices=[0])
    grp = f.h5['Simulations/0000001/Lattice']
    keys = sorted(grp.keys())
    times = np.asarray(f.h5['Simulations/0000001/LatticeTimes'][...][:len(keys)], float)
    frames = np.empty((len(keys), N, N, N), dtype=np.uint8)
    for fi, k in enumerate(keys):
        frames[fi] = (grp[k][...] == P.idx).sum(axis=3).astype(np.uint8)
    return frames, times, n_part


def part_a():
    out = os.path.join(OUT, 'partA.npz')
    res = {}
    with tempfile.TemporaryDirectory(dir=OUT) as wd:
        for N in N_SWEEP:
            t0 = time.time()
            frames, times, n_part = run(N, wd, seed=0)
            # time-average over the second half (static => valid); 1-D marginal along solver-x (axis 0)
            half = frames[len(frames) // 2:]
            marg = half.sum(axis=(2, 3)).astype(np.float64).mean(axis=0)
            res[N] = (marg, n_part)
            print(f'partA N={N:4d} n_part={n_part} placed={int(frames[0].sum())} max_occ={int(frames.max())} '
                  f'{time.time() - t0:.1f}s', flush=True)
            del frames
    np.savez(out, N_SWEEP=np.array(N_SWEEP), L=L, K=K, SIGMA=SIGMA, PPS=PPS, PPS_BUILD=PPS_BUILD,
             **{f'marg_{N}': res[N][0] for N in res}, **{f'npart_{N}': res[N][1] for N in res})
    print('wrote', out)


def replicas():
    with tempfile.TemporaryDirectory(dir=OUT) as wd:
        for seed in range(N_REPLICATES):
            path = os.path.join(OUT, f'rep_s{seed}.h5')
            if os.path.exists(path):
                print('exists', path); continue
            t0 = time.time()
            frames, times, n_part = run(N_REP, wd, seed=seed)
            with h5py.File(path + '.tmp', 'w') as fo:
                fo.attrs.update(dict(seed=seed, L=L, K=K, D=D, UNIT=UNIT, C_STAB=C_STAB, PPS=PPS))
                g = fo.create_group(f'N{N_REP}')
                g.attrs['n_part'] = n_part; g.attrs['h'] = L / N_REP
                g.create_dataset('frames', data=frames, chunks=(1, N_REP, N_REP, N_REP),
                                 compression='gzip', compression_opts=4, shuffle=True)
                g.create_dataset('times', data=times)
            os.replace(path + '.tmp', path)
            print(f'replica seed={seed} n_part={n_part} {time.time() - t0:.1f}s -> {path}', flush=True)
            del frames


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    print(f'build site width {PPS_BUILD}; n_part from PPS = {PPS}', flush=True)
    if what in ('partA', 'all'):
        part_a()
    if what in ('replicas', 'all'):
        replicas()
