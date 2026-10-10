"""WM6 / Figure 4: the two sweeps that test the TEMPORAL flatness condition.

  (a) FREQUENCY response, omega/Dk in OM_RATIOS at fixed tau = tau0. Tests the analytic NESS
      itself: amplitude Dk/sqrt((Dk)^2+omega^2) and lag arctan(omega/Dk). It CANNOT drive
      eps_tau^(1) to unity -- the response attenuates as 1/omega, and by the time
      eps_tau^(1)(2 sigma) reaches 1 (near omega = 30 Dk) the amplitude is 0.67 voxels and the
      center-of-mass fit is shot-noise dominated.
  (b) TIMESTEP response, tau = mult * tau0 at fixed omega = OM_FIXED * Dk. Holds the amplitude
      at 6.3 voxels while carrying eps_tau^(1)(2 sigma) from 0.17 to 0.83, and keeps
      eps_samp = C_STAB * mult <= 1/2. This is the sweep that isolates the condition, because h
      and the physics are both fixed and only tau moves. mult tops out at 5: that is
      eps_samp = 0.50 EXACTLY, the edge of the sampling bound. Do not extend it.

Cost: a run is a fixed number of drive periods, so wallclock scales as 1/omega. omega = 0.1 Dk is
~47k steps at tau0; omega = 10 Dk is ~470. The low-omega end dominates.

Run inside 4d-cell-numba:lm2.6 with the repo at /lm.  Env: WM6_SMOKE=1 for a single cheap run,
WM6_OUT to redirect the npz, WM6_GPU to pick the device.
"""
import os, sys, time, tempfile
# Was a hard pin to /lm/build_container. Resolved through the shared helper now, so the build in use
# is reported rather than assumed; set LM_BUILD=/lm/build_container to restore the old binary
# deliberately. See testing/lm_build.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lm_build import resolve_lm_build, report_imported_lm
resolve_lm_build()
import numpy as np
import lm
report_imported_lm()
from jLM.RDME import Sim as RDMESim
from jLM import Solvers
assert hasattr(lm, 'DriftMpdRdmeSolver')

# ---- physics, identical to the harmonic_trap.ipynb setup cell -----------------
L      = 256.0
UNIT   = 1e-8
D      = 1e-12
K      = 0.01
SIGMA  = 1.0 / np.sqrt(K)
DK     = D * K / UNIT**2
C_STAB = 0.10
PPS    = lm.getCompiledLatticeMaxOccupancy()
C_AMP  = 40.0                      # well-center oscillation amplitude (box units)

N_F       = 128                    # h = 2, held fixed in BOTH sweeps
OM_RATIOS = [0.1, 0.3, 1.0, 3.0, 10.0]
TAU_MULT  = [1, 2, 4, 5]           # x tau0; 5 -> eps_samp = 0.50, the bound
OM_FIXED  = 3.0
N_PER     = 3.0
N_FRAMES  = 120

SMOKE = os.environ.get('WM6_SMOKE', '') == '1'
GPU   = int(os.environ.get('WM6_GPU', '0'))
MODE  = os.environ.get('WM6_MODE', 'both')          # both | freq | tstep
N_F   = int(os.environ.get('WM6_N', str(N_F)))
# Absolute timesteps in microseconds. Parameterize the timestep sweep by ABSOLUTE tau, not by
# multiples of tau0: tau0 = C_STAB h^2 UNIT^2 / D depends on h, so multiples of it would move
# eps_tau when h changes, and the whole point of varying h is to hold eps_tau fixed while
# eps_samp = D tau / h^2 shrinks. That separation is what tells the two failure modes apart.
_taus = os.environ.get('WM6_TAUS_US', '')
# Field-refresh interval, in solver steps. The potential is uploaded by the hook and then held
# FIXED until the next hook, so the quantity that enters the temporal flatness condition is
# tau_hook = HOOK * tau, not tau. Default (0) keeps the notebook's adaptive choice.
HOOK  = int(os.environ.get('WM6_HOOK', '0'))
OM_ONE = os.environ.get('WM6_OMEGA_ONE', '')      # single-point freq run, omega/Dk
N_FRAMES = int(os.environ.get('WM6_FRAMES', str(N_FRAMES)))
SEEDS = [int(x) for x in os.environ.get('WM6_SEEDS', '0').split(',')]
OUT   = os.environ.get('WM6_OUT', '/ftsz/studies/lmrddme_validation/mpd_crddme/figures/wm6_omega_tau_sweep.npz')

h_F   = L / N_F
tau0  = C_STAB * (h_F * UNIT)**2 / D
TAUS  = ([float(x) * 1e-6 for x in _taus.split(',')] if _taus
         else [m * tau0 for m in TAU_MULT])
ax_f  = ((np.arange(N_F) - N_F / 2.0) * h_F).astype(np.float64)
ctr_f = N_F / 2.0
print('N=%d  h=%g box units  tau0=%.1f us  Dk=%g/s  sigma=%g  PPS=%d  gpu=%d  mode=%s'
      % (N_F, h_F, tau0 * 1e6, DK, SIGMA, PPS, GPU, MODE))
print('taus (us): %s   eps_samp: %s'
      % (', '.join('%.0f' % (t * 1e6) for t in TAUS),
         ', '.join('%.3f' % (D * t / (h_F * UNIT)**2) for t in TAUS)))


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


def eps_tau1(tau, omega, r):
    """First-order temporal flatness at distance r from the well center, worst phase, along the
    drag diagonal: sqrt(3) tau k c omega r   (SI Eq. draggedHarmonicTemporalEps)."""
    return np.sqrt(3.0) * tau * K * C_AMP * omega * r


def run_drag(omega, tau, n_per=N_PER, seed=0):
    """One dragged-harmonic run. Returns (amplitude, lag_deg, mean_width, n_steps, n_frames)
    from a least-squares fit of COM(t) = a sin(wt) + b cos(wt) over the last two drive periods."""
    class Well(object):
        def __init__(self, *a, **k):
            super(Well, self).__init__()
            self.calls = 0

        def hookSimulation(self, t, lattice):
            s = C_AMP * np.sin(omega * t)
            o = ax_f - s
            phi = 0.5 * K * (o[:, None, None]**2 + o[None, :, None]**2 + o[None, None, :]**2)
            self.setBasisField(0, np.ascontiguousarray(phi.ravel(), dtype=np.float64))
            self.calls += 1
            return 1

    with tempfile.TemporaryDirectory() as wd:
        sim = RDMESim(name='fpS', filename=os.path.join(wd, 'fpS.lm'),
                      dimensions=[N_F, N_F, N_F], latticeSpacing=h_F * UNIT,
                      regionName='cyt', latticeType='Byte', dt=tau)
        P = sim.species('P')
        sim.diffusionConst('D', D)
        sim.transitionRate(None, None, None, sim.diffusionZero)
        sim.transitionRate(P, sim.region('cyt'), sim.region('cyt'), sim.dc.D)
        sigv = SIGMA / h_F
        n_part = int(np.clip(0.25 * PPS * (2 * np.pi)**1.5 * sigv**3, 2000, 300000))
        seed_gaussian(sim, P, N_F, n_part, np.random.default_rng(seed))
        sim.simulationTime = n_per * 2 * np.pi / omega
        total = int(round(sim.simulationTime / tau))
        sim.latticeWriteInterval = max(1, total // N_FRAMES)
        sim.speciesWriteInterval = sim.latticeWriteInterval
        sim.hookInterval = HOOK if HOOK > 0 else max(1, total // (int(n_per) * 400))
        sim.finalize()
        Sol = Solvers.makeSolver(lm.DriftMpdRdmeSolver, Well)
        solver = Sol()
        solver.setPotentialBasis(np.zeros(N_F**3, np.float32), np.zeros(N_F**3, np.float32))
        solver.setSpeciesCoupling(0, 1.0, 0.0)
        f = sim.run(solver=solver, cudaDevices=[GPU])
        grp = f.h5['Simulations/0000001/Lattice']
        ts = f.h5['Simulations/0000001/LatticeTimes'][...]
        xs = np.arange(N_F)
        T, COM, WID = [], [], []
        for j, key in enumerate(sorted(grp.keys())):
            cnt = (grp[key][...] == P.idx).sum(axis=3)
            cs, ws = [], []
            for axn in (0, 1, 2):
                per = cnt.sum(axis=tuple(a for a in (0, 1, 2) if a != axn)).astype(float)
                m = (xs * per).sum() / per.sum()
                cs.append(m)
                ws.append(np.sqrt((per * (xs - m)**2).sum() / per.sum()))
            T.append(ts[j]); COM.append((np.mean(cs) - ctr_f) * h_F); WID.append(np.mean(ws) * h_F)
    T, COM, WID = map(np.array, (T, COM, WID))
    msk = T > T.max() - 2 * 2 * np.pi / omega
    M = np.c_[np.sin(omega * T[msk]), np.cos(omega * T[msk])]
    coef = np.linalg.lstsq(M, COM[msk], rcond=None)[0]
    amp = float(np.hypot(coef[0], coef[1]))
    lag = float(np.degrees(np.arctan2(-coef[1], coef[0])))
    return amp, (lag if lag >= 0 else lag + 180), float(WID.mean()), total, int(msk.sum())


def record(omega, tau, tag, seed=0):
    t0 = time.time()
    amp, lag, wid, total, nfit = run_drag(omega, tau, seed=seed)
    wall = time.time() - t0
    a_an = DK / np.hypot(DK, omega) * C_AMP
    d_an = np.degrees(np.arctan2(omega, DK))
    et = eps_tau1(tau, omega, 2 * SIGMA)
    print('  %-8s s=%-3d omega/Dk=%6.2f tau=%7.1fus eps_samp=%.3f eps_tau=%.3f | A=%8.3f (an %8.3f) '
          'lag=%6.2f (an %6.2f) w=%6.2f | steps=%d fitpts=%d frames=%d | WALL %6.1fs'
          % (tag, seed, omega / DK, tau * 1e6, C_STAB * tau / tau0, et, amp, a_an, lag, d_an, wid,
             total, nfit, N_FRAMES, wall))
    sys.stdout.flush()
    return (omega / DK, tau, amp, a_an, lag, d_an, wid, et, seed, wall)


if SMOKE:
    print('--- SMOKE: single cheap run ---')
    record(10.0 * DK, tau0, 'smoke')
    raise SystemExit(0)

rows_f, rows_t = [], []
if OM_ONE:
    r = float(OM_ONE)
    print('--- single frequency point, omega = %g Dk, hookInterval = %s ---'
          % (r, HOOK if HOOK > 0 else 'adaptive'))
    rows_f.append(record(r * DK, TAUS[0] if _taus else tau0, 'freq1'))
    np.savez(OUT, freq=np.array(rows_f), tstep=np.zeros((0, 8)),
             cols=np.array(['omega_over_Dk', 'tau_s', 'amp', 'amp_analytic', 'lag_deg',
                            'lag_deg_analytic', 'width', 'eps_tau1_2sigma']),
             meta=np.array([N_F, h_F, tau0, DK, SIGMA, C_AMP, C_STAB, OM_FIXED]))
    print('-> %s' % OUT); raise SystemExit(0)
if MODE in ('both', 'freq'):
    print('--- (a) frequency response, tau = %.0f us ---' % (tau0 * 1e6))
    for r in OM_RATIOS:
        for sd in SEEDS:
            rows_f.append(record(r * DK, tau0, 'freq', seed=sd))
if MODE in ('both', 'tstep'):
    print('--- (b) timestep response, omega = %g Dk ---' % OM_FIXED)
    for tau in TAUS:
        for sd in SEEDS:
            rows_t.append(record(OM_FIXED * DK, tau, 'tstep', seed=sd))

np.savez(OUT, freq=np.array(rows_f), tstep=np.array(rows_t),
         cols=np.array(['omega_over_Dk', 'tau_s', 'amp', 'amp_analytic',
                        'lag_deg', 'lag_deg_analytic', 'width', 'eps_tau1_2sigma',
                        'seed', 'wall_s']),
         meta=np.array([N_F, h_F, tau0, DK, SIGMA, C_AMP, C_STAB, OM_FIXED]))
print('-> %s' % OUT)
