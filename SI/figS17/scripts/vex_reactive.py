"""MPD-RDDME with a reversible bimolecular reaction between hard spheres:
A + B <=> C, with sigma_C set so the reaction conserves excluded volume in the dilute limit.

WHAT THIS MEASURES
    The apparent equilibrium constant K_eff = <N_C> V / (<N_A><N_B>) against packing fraction. In
    an ideal gas it must equal the ideal-gas constant K_id = (c_on/c_off) V_sub built into the
    rates. Any departure is the excess free energy entering the reaction acceptance: the forward
    step destroys two excluded volumes and creates one, so at finite density the two directions are
    no longer balanced by the rate ratio alone.

    Running the same model with the hard core disarmed is the control. If K_eff/K_id stays at one
    there, the shift is the exclusion model and not the reaction machinery.

RATE CONVENTION
    The bimolecular rate is specified per SYSTEM volume, not per subvolume:

        K_id = (c_on / c_off) * V_box,      V_box = M * V_sub

    Assuming V_sub instead is wrong by the subvolume count, here a factor of 32768. Four ideal-gas
    runs fix the relation to better than 0.5% (c_on = 0.0008, 0.0012, 0.0016, 0.0024 give K_eff =
    131.2, 196.6, 261.5, 392.2 nm^3 against 131.1, 196.6, 262.1, 393.2 predicted). Rates far above
    this also saturate -- a nominal K of 6.5e6 nm^3 drives the reaction essentially to completion
    and the measured constant stops tracking the nominal one, which is what makes a single
    high-rate calibration point misleading.

  python vex_reactive.py --nper 3.1 --gpu 0 --out rx_out/lat_n3.1.npz
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

NM_PER_VOXEL = float(os.environ.get('VEX_H_NM', 10.0))
V_NM3 = NM_PER_VOXEL ** 3
N = 32
DT_S = 50e-6
D_M2S = 1.0e-13


def bFex_mix(counts, diams, V=V_NM3):
    """beta*F_ex of a hard-sphere mixture, White Bear / BMCSL."""
    n = np.asarray(counts, float); d = np.asarray(diams, float)
    if n.sum() <= 0 or (d <= 0).all():
        return 0.0
    z = [float(np.sum(n * d ** k)) * math.pi / (6.0 * V) for k in range(4)]
    z0, z1, z2, z3 = z
    if z3 <= 0:
        return 0.0
    if z3 >= 1.0:
        return float('inf')
    om = 1.0 - z3; ln = math.log(om)
    return V * (-(6 / math.pi) * z0 * ln + (18 / math.pi) * z1 * z2 / om
                + (6 / math.pi) * (z2 ** 3 / z3 ** 2) * (z3 / om ** 2 + ln))


def bmcsl_ratio(counts, diams, V=V_NM3):
    """K_eff/K_id predicted by the mixture functional: exp[-(mu_C - mu_A - mu_B)], with each
    chemical potential the discrete increment F(N+1) - F(N), which is what a reaction that adds or
    removes one whole particle actually feels."""
    mu = []
    base = np.asarray(counts, float)
    for i in range(3):
        hi = base.copy(); hi[i] += 1.0
        mu.append(bFex_mix(hi, diams, V) - bFex_mix(base, diams, V))
    return math.exp(-(mu[2] - mu[0] - mu[1]))


def build(path, nA, nB, nC, c_on, c_off, n_steps, n_frames, seed=None, dt=DT_S,
          dims=(N, N, N)):
    # pyLM discretizes addParticles placements HOST-side with the interpreter's global RNGs at
    # build time; unseeded, the .lm file itself differs between same-seed runs (frame 0 differs).
    if seed is not None:
        import random as _random
        _random.seed(int(seed)); np.random.seed(int(seed) & 0x7FFFFFFF)
    from pyLM import RDME
    from pyLM.units import nm, micron, microsecond
    ux, uy, uz = (d * NM_PER_VOXEL * 1e-3 for d in dims)
    sim = RDME.RDMESimulation(dimensions=micron(ux, uy, uz),
                              spacing=nm(NM_PER_VOXEL))
    sim.defineSpecies(['A', 'B', 'C'])
    reg = sim.modifyRegion('default')
    reg.setDefaultDiffusionRate(D_M2S)
    reg.addReaction(reactant=('A', 'B'), product='C', rate=c_on)
    reg.addReaction(reactant='C', product=('A', 'B'), rate=c_off)
    for s, n_ in (('A', nA), ('B', nB), ('C', nC)):
        if n_:
            sim.addParticles(species=s, region='default', count=int(n_))
    sim.setTimestep(microsecond(dt * 1e6))
    per_frame = max(1, n_steps // n_frames)
    sim.setWriteInterval(per_frame)
    sim.setLatticeWriteInterval(per_frame)
    sim.setSimulationTime(n_steps * dt)
    if seed is not None:
        sim.setRandomSeed(int(seed))   # independent replicates need independent streams
    sim.save(path)
    if seed is not None:
        # pyLM's setRandomSeed does not reach the initial-placement RNG; the file parameter does
        # (same fix as the FtsZ harness). Without it frame 0 differs between same-seed runs.
        f = __import__('lm').SimulationFile(path)
        f.setParameter('seed', str(int(seed)))
        f.close()
    return per_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sigma', default='3,3,3.78', help='sigma_A,sigma_B,sigma_C in nm')
    ap.add_argument('--nper-c', type=float, default=0.0,
                    help='initial C per subvolume; nonzero only for fixed-composition runs')
    ap.add_argument('--nper', type=float, required=True,
                    help='initial A (and B) per subvolume; C starts empty')
    ap.add_argument('--K', type=float, default=200.0, help='ideal-gas constant, nm^3')
    ap.add_argument('--c-off', type=float, default=200.0, help='dissociation rate, 1/s')
    ap.add_argument('--c-on', type=float, default=None,
                    help='association rate, 1/s; overrides the value derived from --K')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--tail', type=float, default=0.5, help='fraction of frames averaged')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--no-exclusion', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--gamma', default='0,0,0',
                    help='per-species amplitude in kT of phi_s = gamma_s sin(2 pi x/lambda)')
    ap.add_argument('--phase', type=float, default=0.0,
                    help='phase offset in radians of the sinusoid relative to the subvolume grid')
    ap.add_argument('--wavelength', type=float, default=40.0,
                    help='potential wavelength in nm; must divide the box and match the BD runs')
    ap.add_argument('--dt', type=float, default=None,
                    help='timestep in s; the default 50 us resolves only ~40 steps over the '
                         'chemical relaxation, too coarse for a time-resolved comparison')
    ap.add_argument('--tau-max', type=float, default=None,
                    help='run length in units of tau = D t / h^2, overrides --steps')
    ap.add_argument('--timeseries', action='store_true',
                    help='store the per-frame occupancy of C and the species counts')
    ap.add_argument('--dims', default='32,32,32',
                    help='lattice sites X,Y,Z; the MPD kernel requires X %% 32 == 0 and '
                         'Y, Z %% 8 == 0. The field runs along X. Default matches the old '
                         'hardcoded 32^3 exactly.')
    ap.add_argument('--baseline-kernel', action='store_true',
                    help='run the stock MpdRdmeSolver instead of DriftMpdRdmeSolver: no field, '
                         'no exclusion, no drift machinery. The distributions are WRONG for the '
                         'driven/crowded problem; benchmark-only, as the stock-LM throughput '
                         'reference.')
    ap.add_argument('--bench', action='store_true',
                    help='timing mode: no timeseries, no profile analysis, minimal frame '
                         'writes; reports the wall time of the solver call alone so a '
                         'driver can difference two step counts')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()

    sig = [float(x) for x in A.sigma.split(',')]
    gam = [float(x) for x in A.gamma.split(',')]
    dims = tuple(int(x) for x in A.dims.split(','))
    NX, NY, NZ = dims
    if NX % 32 or NY % 8 or NZ % 8:
        raise SystemExit('--dims: MPD kernels require X %% 32 == 0 and Y, Z %% 8 == 0 (got %s)'
                         % (dims,))
    # VEX_ALLOW_DIMS=1: run a non-32^3 lattice with frames (for frame-hash identity gates on larger lattices, e.g. the
    # ERA Tier-0 64^3 sparse config); the 32^3-specific profile/timeseries analysis is then skipped after the run.
    allow_dims = bool(int(os.environ.get('VEX_ALLOW_DIMS', '0')))
    if dims != (N, N, N) and not A.bench and not allow_dims:
        raise SystemExit('the profile/timeseries analysis assumes the cubic 32^3 lattice; '
                         'non-default --dims is supported in --bench mode only (or set VEX_ALLOW_DIMS=1 to run '
                         'and skip the analysis)')
    M = NX * NY * NZ
    c_on = A.c_on if A.c_on is not None else A.c_off * A.K / (V_NM3 * M)
    nA0 = int(round(A.nper * M)); nB0 = nA0
    nC0 = int(round(A.nper_c * M))          # 0 unless a fixed-composition (non-reacting) run
    print('sigma=%s nper=%.3g  c_on=%.4g /s  c_off=%.4g /s  -> K_id=%.4g nm^3'
          % (sig, A.nper, c_on, A.c_off, c_on / A.c_off * V_NM3 * M)); sys.stdout.flush()

    dt_s = A.dt if A.dt else DT_S
    D_nm2s = D_M2S * 1e18
    steps = A.steps if A.tau_max is None else int(math.ceil(
        A.tau_max * NM_PER_VOXEL ** 2 / D_nm2s / dt_s))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        # VEX_REUSE_LM: run from a prebuilt .lm file (copied, since the solver writes into it).
        # Isolates dynamics determinism from build-time placement randomness for gate validation.
        reuse = os.environ.get('VEX_REUSE_LM', '')
        path = os.path.join(td, 'rx.lm')
        frames = 1 if A.bench else A.frames
        if reuse and os.path.exists(reuse):
            import shutil
            shutil.copy(reuse, path)
            per_frame = max(1, steps // frames)
        else:
            per_frame = build(path, nA0, nB0, nC0, c_on, A.c_off, steps, frames, seed=A.seed,
                              dt=dt_s, dims=dims)
            if reuse:
                import shutil
                shutil.copy(path, reuse)
        solver = lm.MpdRdmeSolver() if A.baseline_kernel else lm.DriftMpdRdmeSolver()
        Nxyz = M
        # linear ramp along x in the notebook convention: psi0 = x/N - 1/2 spans [-1/2, 1/2] and
        # phi_s = gamma_s psi0, so gamma_s is the drop across the box in kT. The box is periodic,
        # so the ramp is a sawtooth and the seam subvolume carries the whole return; it is excluded
        # from the analysis rather than smoothed away.
        # Sinusoid of fixed wavelength, not a box-spanning ramp: a ramp equilibrates on
        # L^2/(D gamma), which differs by 64x between the 40 nm Brownian box and this 320 nm one,
        # so the two would never be at the same stage of relaxation at the same tau. A fixed
        # wavelength relaxes at D k^2, identical in both, and is continuous at the wrap.
        # Broadcast as the demo notebooks do: broadcast_to fills the LAST index, which is the axis
        # the kernel treats as x.
        # At lambda = 4h the lattice samples the sinusoid four times per period, so it sees only
        # the three values 0 and +/- gamma. A phase offset slides the sample points along the same
        # field: each run is a valid equilibrium in its own right, and runs at several phases trace
        # out more of n(Phi) without changing lambda, h, or the box.
        nlam = A.wavelength / NM_PER_VOXEL
        line = np.sin(2.0 * np.pi * np.arange(NX, dtype=np.float32) / nlam
                      + A.phase).astype(np.float32)
        if not A.baseline_kernel:
            psi0 = np.broadcast_to(line, (NZ * NY, NX)).astype(np.float32).ravel(order='C')
            solver.setPotentialBasis(np.ascontiguousarray(psi0), np.zeros(Nxyz, np.float32))
            for i in range(3):
                solver.setSpeciesCoupling(i, float(gam[i]), 0.0)
            if not A.no_exclusion:
                solver.setSpeciesHardSphereDiameters(np.array(sig, np.float64), float(V_NM3))
                # VEX_LAMBDA_HC: hard-core prefactor override (default 1.0). Used by the ERA Tier-1 gate's power
                # test — a deliberate x% error in the exclusion work must be caught by the statistical comparison.
                solver.setHardCoreExclusion(float(os.environ.get('VEX_LAMBDA_HC', '1.0')), 2)
            solver.setHopIntegration(1)
        # Perf gates (byte-identical when the word high-water-mark invariant holds; verified via
        # getLatticeWordBoundViolations below). Default off to keep prior runs' provenance exact.
        if int(os.environ.get('VEX_WORD_BOUND', '0')) and not A.baseline_kernel:
            solver.setLatticeWordBound(1)
            solver.setLatticeTransportBound(1)
        # Species-memoized drift probabilities: byte-identical (pure reuse), default off.
        if int(os.environ.get('VEX_SPECIES_MEMO', '0')) and not A.baseline_kernel:
            solver.setDriftSpeciesMemo(1)
        # Factorized reaction propensity: same sum re-associated (statistically gated), default off.
        if int(os.environ.get('VEX_RXN_FACTOR', '0')) and not A.baseline_kernel:
            solver.setRxnCountFactor(1)
        # Adaptive tau is a feedback loop and a suspected source of run-to-run nondeterminism;
        # freezing it (VEX_ADAPTIVE_TAU=0) enables bit-exact A/B validation of perf gates.
        if int(os.environ.get('VEX_ADAPTIVE_TAU', '1')) and not A.baseline_kernel:
            solver.setAdaptiveTau(1, 0.2, 0.05)
        # VEX_ORDER=perm: the step's operators run in a timestep-seeded random order (rddme/StepOrder.cu); the default
        # fixed order calls nothing, so a build without the setter runs unchanged. A build that lacks it under perm stops.
        if os.environ.get('VEX_ORDER', 'fixed') == 'perm' and not A.baseline_kernel:
            if not hasattr(solver, 'setStepOrderRandomize'):
                sys.exit('VEX_ORDER=perm needs a build with setStepOrderRandomize (rddme/StepOrder.cu)')
            solver.setStepOrderRandomize(1)
        if not A.baseline_kernel:
            solver.validateStability()
        t_solver = time.time()
        lm.runSolver(path, 1, solver, [A.gpu], 0)
        wall_solver = time.time() - t_solver
        # VEX_KEEP_LM: copy the finished .lm (all lattice frames) out of the tempdir so a gate harness
        # can hash the raw lattice bytes frame by frame (byte identity is on the bytes, not on counts).
        keep = os.environ.get('VEX_KEEP_LM', '')
        if keep:
            import shutil
            os.makedirs(os.path.dirname(keep) or '.', exist_ok=True)
            shutil.copy(path, keep)
        if int(os.environ.get('VEX_WORD_BOUND', '0')) and not A.baseline_kernel:
            v = int(solver.getLatticeWordBoundViolations())
            print('WORDMARK max=%d violations=%d' % (int(solver.getLatticeMaxWord()), v))
            sys.stdout.flush()
            if v and not int(os.environ.get('VEX_WORD_BOUND_TOLERATE', '0')):
                raise SystemExit('word-bound invariant VIOLATED %d times: results unsafe' % v)
        if A.bench:
            tau_scale = 1.0 if A.baseline_kernel else float(solver.getTauScale())
            meta = dict(sigma=sig, nper=A.nper, seed=A.seed, gamma=gam,
                        wavelength=A.wavelength, no_exclusion=bool(A.no_exclusion),
                        dims=list(dims), M=M, nA0=nA0, dt_s=dt_s, tau_scale=tau_scale,
                        steps=steps, wall_solver_s=wall_solver,
                        sim_s=steps * dt_s * tau_scale, wall_total_s=time.time() - t0)
            os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
            np.savez_compressed(A.out, meta=json.dumps(meta))
            print('BENCH dims=%s M=%d nA0=%d steps=%d wall_solver=%.3fs tau_scale=%.4g '
                  'sim_s=%.4g' % (dims, M, nA0, steps, wall_solver, tau_scale,
                                  meta['sim_s'])); sys.stdout.flush()
            return
        if dims != (N, N, N):
            print('ANALYSIS SKIPPED: non-32^3 dims under VEX_ALLOW_DIMS (lattice frames kept via VEX_KEEP_LM)')
            sys.stdout.flush()
            return
        # adaptive tau rescales the physical step, so the clock is dt*tau_scale, not dt.
        # Omitting this stretches the tau axis by 1/tau_scale -- the diffusive script
        # vex_relax.py folds it in and this one did not.
        tau_scale = 1.0 if A.baseline_kernel else float(solver.getTauScale())
        with h5py.File(path, 'r') as f:
            g = f['/Simulations/0000001/Lattice']
            names = sorted(g.keys())
            series, occC, occA = [], [], []
            if A.timeseries:
                for nm_ in names:
                    l = g[nm_][...]
                    series.append([int((l == i + 1).sum()) for i in range(3)])
                    occC.append((l == 3).sum(axis=-1).ravel().astype(np.int16))
                    occA.append((l == 1).sum(axis=-1).ravel().astype(np.int16))
            tail = names[int(len(names) * A.tail):]
            cnt = np.zeros(3)
            # Occupancy resolved along the ramp axis. Under a ramp the global K_eff is meaningless
            # -- the equilibrium constant is position dependent -- so the comparable observable is
            # the profile. psi0 was built by broadcasting `line` along the FIRST axis of a C-ordered
            # (N,N,N) array, so averaging axes 1 and 2 of the same reshape recovers it; the
            # ideal-gas control checks that convention, since a wrong axis would flatten the
            # profile to slope zero instead of the expected -gamma.
            # Profile along each of the three lattice axes. Which one carries the ramp depends on
            # how the solver's flat site index maps onto the HDF5 lattice shape, and guessing it
            # wrong averages over the ramp and returns a flat profile. Storing all three settles it
            # from the data: the ideal-gas control must show slope -gamma on exactly one axis.
            prof3 = np.zeros((3, 3, N))
            for nm_ in tail:
                l = g[nm_][...]
                for i in range(3):
                    occ = (l == i + 1).sum(axis=-1).reshape(N, N, N)
                    cnt[i] += occ.sum()
                    for ax in range(3):
                        prof3[i, ax] += occ.mean(axis=tuple(a for a in range(3) if a != ax))
            cnt /= len(tail)
            prof3 /= len(tail)
    NA_, NB_, NC_ = cnt
    Vbox = M * V_NM3
    K_eff = NC_ * Vbox / (NA_ * NB_) if NA_ > 0 and NB_ > 0 else float('nan')
    per = cnt / M
    eta = (math.pi / 6) * sum(per[i] * sig[i] ** 3 for i in range(3)) / V_NM3
    pred = bmcsl_ratio(per, sig)
    meta = dict(sigma=sig, nper=A.nper, seed=A.seed, gamma=gam, wavelength=A.wavelength, phase=float(A.phase), no_exclusion=bool(A.no_exclusion), K_id=A.K,
                c_on=c_on, c_off=A.c_off, N=[float(x) for x in cnt], per_subvolume=list(per),
                eta=float(eta), K_eff=float(K_eff), ratio_meas=float(K_eff / A.K),
                ratio_bmcsl=float(pred), steps=A.steps, wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    # Record the field ACTUALLY used, not a formula for it: this line was hardcoded to the
    # old linear ramp and silently kept reporting a ramp after the field became a sinusoid.
    extra = dict(prof3=prof3, psi0_line=np.asarray(line, dtype=np.float64))
    if A.timeseries:
        extra.update(tau=np.arange(len(series)) * per_frame * dt_s * tau_scale
                         * D_nm2s / NM_PER_VOXEL ** 2,
                     counts=np.array(series), occC=np.array(occC),
                     occA=np.array(occA))
    np.savez_compressed(A.out, meta=json.dumps(meta), **extra)
    print('  eta=%.4f  <n>=(%.3f,%.3f,%.3f)  K_eff/K_id=%.4f  BMCSL %.4f  (%.0fs)'
          % (eta, per[0], per[1], per[2], meta['ratio_meas'], pred, meta['wall_s']))


if __name__ == '__main__':
    main()
