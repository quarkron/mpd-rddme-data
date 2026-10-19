"""MPD-RDDME subvolume occupancy for a BINARY hard-sphere mixture, one size ratio per process.

The one-species sweep (vex_sigma_sweep.py) cannot test the mixture terms of the White Bear
functional: with a single diameter the moment ratios lambda = M1 M2 / M3 and gamma = M1^3 / M3 are
identically 1, so those terms are never exercised. This runs the same measurement for two species
at a fixed TOTAL packing fraction split EQUALLY BY VOLUME, N_A sigma_A^3 = N_B sigma_B^3, sweeping
the size ratio, and persists the per-species marginals and the joint distribution.

The analytic companion is the grand-canonical lattice gas built on the MIXTURE White Bear
functional, with two fugacities fitted to the two target means -- the direct generalization of
what the one-species script does with one fugacity.

  python vex_twospecies.py --ratio 2.0 --gpu 0 --out vex_out2/two_r2.npz
"""
import argparse, json, time

import os, sys, tempfile, math
import numpy as np
import h5py

# Build resolution lives in testing/lm_build.py. It is NOT inlined here: the loop this
# replaced inserted every candidate at sys.path[0], which reversed its own priority order and
# silently loaded an eight-week-old binary for an entire campaign. See that module's docstring.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lm_build import resolve_lm_build, report_imported_lm
resolve_lm_build()
import lm
report_imported_lm()
assert hasattr(lm, 'DriftMpdRdmeSolver'), 'DriftMpdRdmeSolver missing from lm build'
assert hasattr(lm.DriftMpdRdmeSolver(), 'setSpeciesHardSphereDiameters'), \
    'this LM build predates Stage C2 (White Bear) -- rebuild with the WHITE_BEAR form'


# ---- physics ------------------------------------------------------------------
NM_PER_VOXEL = float(os.environ.get('VEX_H_NM', 10.0))
V_NM3        = NM_PER_VOXEL ** 3
N            = 32                    # lattice edge (MUST be a multiple of 32)
DT_S         = 50e-6
N_STEPS      = 40000
N_FRAMES     = 6


def bFex_mix(n_vec, sig_vec, V=V_NM3):
    """beta*F_ex of a MIXTURE of hard spheres in one subvolume, White Bear.

    Identical in form to the one-species bFex, with the scaled weighted densities summed over
    species, x_k = (pi/6V) sum_i n_i sigma_i^k. The single-species function is the special case
    of one term in each sum, so a disagreement between the two at equal diameters would be a bug
    in one of them."""
    n_vec = np.atleast_1d(np.asarray(n_vec, float))
    s = np.atleast_1d(np.asarray(sig_vec, float))
    if n_vec.sum() <= 0 or (s <= 0).all():
        return 0.0
    c = math.pi / (6.0 * V)
    x0 = c * n_vec.sum()
    x1 = c * float((n_vec * s).sum())
    x2 = c * float((n_vec * s ** 2).sum())
    x3 = c * float((n_vec * s ** 3).sum())
    x3 = min(x3, 1.0 - 1e-9)
    if x3 <= 0.0:
        return 0.0
    ip, omx = 1.0 / math.pi, 1.0 - x3
    ln = math.log(omx)
    return V * (-(6 * ip) * x0 * ln + (18 * ip) * x1 * x2 / omx
                + (6 * ip) * (x2 ** 3 / x3 ** 2) * (x3 / omx ** 2 + ln))


def lattice_gas_joint(zA, zB, sig, nmax):
    """P(n_A, n_B) propto zA^nA zB^nB exp(-bFex_mix)/ (nA! nB!). Built in LOG space: the direct
    product overflows well before the occupancies of interest."""
    na = np.arange(nmax + 1)
    logw = np.full((nmax + 1, nmax + 1), -np.inf)
    for i in na:
        for j in na:
            logw[i, j] = (i * math.log(zA) + j * math.log(zB)
                          - bFex_mix([i, j], sig) - math.lgamma(i + 1) - math.lgamma(j + 1))
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def fit_fugacities(target, sig, nmax, iters=200):
    """Two fugacities for two target means. The means are monotone in their own fugacity at fixed
    other, so alternating bisection converges; a joint Newton step is not worth the Jacobian."""
    na = np.arange(nmax + 1)
    z = [1.0, 1.0]
    for _ in range(iters):
        for s in (0, 1):
            lo, hi = 1e-6, 1e3
            for _g in range(80):
                z[s] = hi
                P = lattice_gas_joint(z[0], z[1], sig, nmax)
                m = (na * P.sum(axis=1 - s)).sum()
                if m >= target[s]:
                    break
                hi *= 10.0
            for _b in range(60):
                z[s] = math.sqrt(lo * hi)
                P = lattice_gas_joint(z[0], z[1], sig, nmax)
                m = (na * P.sum(axis=1 - s)).sum()
                if m < target[s]:
                    lo = z[s]
                else:
                    hi = z[s]
            z[s] = math.sqrt(lo * hi)
        P = lattice_gas_joint(z[0], z[1], sig, nmax)
        mA = (na * P.sum(axis=1)).sum(); mB = (na * P.sum(axis=0)).sum()
        if abs(mA - target[0]) < 1e-4 * target[0] and abs(mB - target[1]) < 1e-4 * target[1]:
            break
    return z, P


# ---- simulate -----------------------------------------------------------------
import contextlib


def _build(path, nA, nB, n_steps=None, n_frames=None):
    n_steps = int(n_steps or N_STEPS); n_frames = int(n_frames or N_FRAMES)
    from pyLM import RDME
    from pyLM.units import nm, micron, microsecond
    box_um = N * NM_PER_VOXEL * 1e-3
    sim = RDME.RDMESimulation(dimensions=micron(box_um, box_um, box_um),
                              spacing=nm(NM_PER_VOXEL))
    sim.defineSpecies(['A', 'B'])
    sim.modifyRegion('default').setDefaultDiffusionRate(1.0e-13)
    sim.addParticles(species='A', region='default', count=int(nA))
    sim.addParticles(species='B', region='default', count=int(nB))
    sim.setTimestep(microsecond(50))
    per_frame = max(1, n_steps // n_frames)
    sim.setWriteInterval(per_frame)
    sim.setLatticeWriteInterval(per_frame)
    sim.setSimulationTime(n_steps * DT_S)
    sim.save(path)


@contextlib.contextmanager
def _capture_c_log():
    out = ['']
    so, se = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode='w+b')
    try:
        os.dup2(tmp.fileno(), 1); os.dup2(tmp.fileno(), 2)
        yield out
    finally:
        os.dup2(so, 1); os.dup2(se, 2); os.close(so); os.close(se)
        tmp.flush(); tmp.seek(0); out[0] = tmp.read().decode('utf-8', 'replace'); tmp.close()


_ABORT_MARKERS = ('Too many particle overflows', 'Exception during execution',
                  'Error during execution', 'CUDA error', 'cuda error')


def _run_solver_loud(path, solver, gpu):
    with _capture_c_log() as log:
        lm.runSolver(path, 1, solver, [gpu], 0)
    text = log[0]
    hit = next((m for m in _ABORT_MARKERS if m in text), None)
    if hit:
        raise RuntimeError(
            "\n*** LM SOLVER ABORTED ('" + hit + "') ***\n"
            "The frames written are the un-evolved INITIAL (Poisson) state, so any apparent\n"
            "agreement with the point-particle limit is meaningless. Reduce sigma or the density.")
    n_ovf = text.count('uses of the particle overflow list')
    if n_ovf:
        print("  [WARNING] particle overflow list used in %d window(s): the hop is overdriven; "
              "the distribution may be a non-physical jam, not the true sub-Poissonian." % n_ovf)
    return n_ovf


def run_occupancy(nA, nB, sig, gpu=0, start_frac=0.5, adaptive=True, eps=0.2, tau_init=0.05,
                  n_steps=None, n_frames=None):
    """Returns (P(nA), P(nB), joint, moments dict). Species ids in the HDF5 lattice are
    1-indexed in definition order, so A==1 and B==2 and 0 is an empty slot."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'vex2.lm')
        _build(path, nA, nB, n_steps, n_frames)
        solver = lm.DriftMpdRdmeSolver()
        Nxyz = N * N * N
        solver.setPotentialBasis(np.zeros(Nxyz, np.float32), np.zeros(Nxyz, np.float32))
        solver.setSpeciesCoupling(0, 1.0, 0.0)
        solver.setSpeciesCoupling(1, 1.0, 0.0)
        solver.setSpeciesHardSphereDiameters(np.array(sig, np.float64), float(V_NM3))
        solver.setHardCoreExclusion(1.0, 2)         # lambda=1 (physical), WHITE_BEAR
        solver.setHopIntegration(1)                 # EXACT (Poisson-tau-leap)
        if adaptive:
            solver.setAdaptiveTau(1, float(eps), float(tau_init))
        solver.validateStability()
        n_ovf = _run_solver_loud(path, solver, gpu)
        tau_scale = float(solver.getTauScale()) if adaptive else 1.0
        if adaptive:
            print("  [adaptive tau] settled tau_scale=%.4g, last max(W*tau')=%.3f"
                  % (tau_scale, solver.getLastSumMax()))
        with h5py.File(path, 'r') as f:
            g = f['/Simulations/0000001/Lattice']
            names = sorted(g.keys())
            if len(names) < 2:
                raise RuntimeError('solver wrote %d lattice frame(s) -- run aborted before '
                                   'evolving' % len(names))
            tail = names[int(len(names) * start_frac):]
            max_occ = int(g[tail[0]].shape[-1])
            accA, accB = [], []
            for nm_ in tail:
                lat = g[nm_][...]
                accA.append((lat == 1).sum(axis=-1).ravel())
                accB.append((lat == 2).sum(axis=-1).ravel())
    cA = np.concatenate(accA); cB = np.concatenate(accB)
    PA = np.bincount(cA, minlength=max_occ + 1).astype(float); PA /= PA.sum()
    PB = np.bincount(cB, minlength=max_occ + 1).astype(float); PB /= PB.sum()
    J = np.histogram2d(cA, cB, bins=[np.arange(max_occ + 2), np.arange(max_occ + 2)])[0]
    J /= J.sum()
    return PA, PB, J, cA, cB, max_occ, tau_scale, n_ovf


def moments_of(c):
    c = c.astype(float)
    mean = c.mean(); var = c.var(); sd = math.sqrt(var)
    return dict(mean=float(mean), var=float(var), fano=float(var / mean) if mean > 0 else float('nan'),
                cv=float(sd / mean) if mean > 0 else float('nan'),
                skew=float(((c - mean) ** 3).mean() / sd ** 3) if sd > 0 else float('nan'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ratio', type=float, required=True, help='sigma_B / sigma_A')
    ap.add_argument('--eta', type=float, default=0.20, help='TOTAL packing fraction, held fixed')
    ap.add_argument('--sigma-a', type=float, default=3.0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--nmax-analytic', type=int, default=28)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()

    sig = [A.sigma_a, A.sigma_a * A.ratio]
    # equal volume split at fixed total eta, expressed per subvolume then scaled to the lattice
    nA_per = (3.0 / math.pi) * A.eta * V_NM3 / sig[0] ** 3
    nB_per = nA_per * (sig[0] / sig[1]) ** 3
    NA = int(round(nA_per * N ** 3)); NB = int(round(nB_per * N ** 3))
    eta = (math.pi / 6) * (nA_per * sig[0] ** 3 + nB_per * sig[1] ** 3) / V_NM3
    print('ratio=%.3g sigma=(%.2f,%.2f) eta=%.4f  <nA>=%.3f <nB>=%.3f  N=(%d,%d) on %d^3'
          % (A.ratio, sig[0], sig[1], eta, nA_per, nB_per, NA, NB, N)); sys.stdout.flush()

    t0 = time.time()
    PA, PB, J, cA, cB, max_occ, tau_scale, n_ovf = run_occupancy(NA, NB, sig, gpu=A.gpu,
                                                                 n_steps=A.steps)
    wall = time.time() - t0
    mA = moments_of(cA); mB = moments_of(cB)
    a = cA.astype(float); b = cB.astype(float)
    cov = float(((a - a.mean()) * (b - b.mean())).mean())
    pear = float(cov / (a.std() * b.std())) if a.std() > 0 and b.std() > 0 else float('nan')

    print('  measured  <nA>=%.3f FanoA=%.4f | <nB>=%.3f FanoB=%.4f | cov=%+.4f rho=%+.4f'
          % (mA['mean'], mA['fano'], mB['mean'], mB['fano'], cov, pear)); sys.stdout.flush()

    # analytic mixture lattice gas at the SAME means
    nmax = A.nmax_analytic
    z, Pj = fit_fugacities([nA_per, nB_per], sig, nmax)
    na = np.arange(nmax + 1)
    PAa = Pj.sum(axis=1); PBa = Pj.sum(axis=0)
    mAa = (na * PAa).sum(); vAa = (na ** 2 * PAa).sum() - mAa ** 2
    mBa = (na * PBa).sum(); vBa = (na ** 2 * PBa).sum() - mBa ** 2
    covA = float((na[:, None] * na[None, :] * Pj).sum() - mAa * mBa)
    print('  analytic  <nA>=%.3f FanoA=%.4f | <nB>=%.3f FanoB=%.4f | cov=%+.4f'
          % (mAa, vAa / mAa, mBa, vBa / mBa, covA))

    x = np.asarray([nA_per, nB_per]); x = x / x.sum(); s = np.asarray(sig)
    M1 = (x * s).sum(); M2 = (x * s ** 2).sum(); M3 = (x * s ** 3).sum()
    meta = dict(ratio=A.ratio, sigma=sig, eta_tot=eta, nA_per=nA_per, nB_per=nB_per,
                NA=NA, NB=NB, lattice=N, h_nm=NM_PER_VOXEL, steps=A.steps, gpu=A.gpu,
                wall_s=wall, tau_scale=tau_scale, n_overflow_windows=n_ovf, max_occ=max_occ,
                A=mA, B=mB, cov=cov, pearson=pear, z=z,
                analytic=dict(fanoA=float(vAa / mAa), fanoB=float(vBa / mBa), cov=covA,
                              meanA=float(mAa), meanB=float(mBa)),
                bmcsl=dict(M1=float(M1), M2=float(M2), M3=float(M3),
                           lam=float(M1 * M2 / M3), gam=float(M1 ** 3 / M3)))
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    np.savez_compressed(A.out, PA=PA, PB=PB, J=J, PA_analytic=PAa, PB_analytic=PBa,
                        J_analytic=Pj, meta=json.dumps(meta))
    print('  -> %s  (%.0fs)' % (A.out, wall))


if __name__ == '__main__':
    main()
