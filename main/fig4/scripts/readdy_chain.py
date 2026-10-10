#!/usr/bin/env python3
"""Off-lattice semiflexible chain in ReaDDy: the dynamics reference for the lattice chain.

WHY READDY AND NOT SOMETHING ELSE. It is already this paper's cross-engine comparator, so the
polymer section inherits a reference the manuscript has justified rather than introducing a third
bespoke sampler. More importantly it integrates overdamped Langevin dynamics with independent per
particle noise, which means it is free draining, exactly like a local lattice move set. Comparing
against a reference that carried hydrodynamic interactions would confound a real disagreement with a
Zimm crossover.

WHAT IS MATCHED, AND WHAT IS NOT. The two bending energies have different functional forms, one in
the cosine of the joint angle and one quadratic in the angle itself, and the lattice never samples
angles below thirty five degrees. Matching the microscopic stiffness constant between them would be
meaningless. What is matched instead is four EMERGENT quantities: the bond length, the persistence
length, the single subunit diffusion coefficient and the bead diameter. The persistence length match
is a one dimensional root find on the angle force constant against the exact lattice value rather
than against a fitted one.

Once those four agree there is no remaining freedom in the relaxation spectrum, because Rouse theory
fixes it. So the dynamics comparison downstream of this module is a prediction test and not a fit.

ENERGY CONVENTION. ReaDDy's harmonic potentials carry NO factor of one half: the bond energy is the
force constant times the squared extension, not half of it. Measured bond length variance at a
force constant of two hundred converges to about 0.0027, which is one over twice the force constant
and not one over it. This does not affect the matching, which is done on emergent quantities by root
finding, but it does mean a force constant here is not comparable to a lattice stiffness by eye.

UNITS. Length is in lattice voxels and time is in seconds, so a diffusion coefficient of 0.3 square
micrometres per second at a ten nanometre voxel is three thousand square voxels per second. Keeping
length in voxels means every number here can be compared to a lattice number without conversion,
which is one fewer place for a silent unit error.
"""
import argparse
import os

import numpy as np

os.environ.setdefault('CONDA_PREFIX', '/raid/racda/opt/readdy_env')

VOXEL_NM = 10.0
D1_VOXEL2_PER_S = 3000.0          # 0.3 um^2/s at a 10 nm voxel
BOND_EDGE = float(np.sqrt(2.0))   # the edge-orbit bond length, in voxels
SIGMA = 1.0                       # bead diameter: two subunits cannot share one voxel


def build(n, bond_len=BOND_EDGE, k_angle=1.0, k_bond=200.0, k_rep=200.0,
          sigma=SIGMA, d1=D1_VOXEL2_PER_S, box=None):
    """A single chain of n beads in a periodic box, with bonds, angles and excluded volume.

    The box is sized from the contour length so a chain cannot exclude volume against its own
    periodic image, which is the same requirement the lattice runs have and for the same reason.
    """
    import readdy
    box = box or max(24.0, 3.0 * n * bond_len / np.pi)
    system = readdy.ReactionDiffusionSystem(
        box_size=[box, box, box], periodic_boundary_conditions=[True, True, True],
        unit_system=None)
    system.add_topology_species('M', d1)
    system.topologies.add_type('chain')
    system.topologies.configure_harmonic_bond('M', 'M', force_constant=k_bond, length=bond_len)
    system.topologies.configure_harmonic_angle('M', 'M', 'M', force_constant=k_angle,
                                               equilibrium_angle=np.pi)
    system.potentials.add_harmonic_repulsion('M', 'M', force_constant=k_rep,
                                             interaction_distance=sigma)
    return system, box


def seed_positions(n, bond_len, box, rng, kind='coil'):
    """Starting configuration. A coil start approaches equilibrium from below in stiffness and a
    straight start from above, which is the same two-sided bracket the lattice runs use. A single
    start cannot distinguish an equilibrated measurement from one that stopped early."""
    if kind == 'straight':
        pos = np.array([[i * bond_len, 0.0, 0.0] for i in range(n)])
        pos -= pos.mean(axis=0)
        if np.abs(pos).max() > 0.45 * box:
            pos *= 0.45 * box / np.abs(pos).max()
        return pos
    pos = [np.zeros(3)]
    for _ in range(n - 1):
        for _try in range(500):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            p = pos[-1] + bond_len * v
            if all(np.linalg.norm(p - q) > SIGMA for q in pos[:-1]):
                break
        pos.append(p)
    pos = np.array(pos)
    pos -= pos.mean(axis=0)
    if np.abs(pos).max() > 0.45 * box:
        pos *= 0.45 * box / np.abs(pos).max()
    return pos


def run(n, n_steps, dt, stride, k_angle, bond_len=BOND_EDGE, seed=0, k_bond=200.0,
        k_rep=200.0, d1=D1_VOXEL2_PER_S, box=None, kernel='SingleCPU', start='coil'):
    """Integrate and return the unwrapped bead positions at each recorded frame.

    ReaDDy reports positions folded into the periodic box. A chain that straddles a face would then
    look folded, so positions are unwrapped along the backbone using the bond length as the scale,
    exactly as the lattice reader unwraps along the successor walk.
    """
    import readdy
    system, box = build(n, bond_len, k_angle, k_bond, k_rep, d1=d1, box=box)
    sim = system.simulation(kernel=kernel)
    rng = np.random.default_rng(seed)
    init = seed_positions(n, bond_len, box, rng, start)
    top = sim.add_topology('chain', ['M'] * n, init)
    for i in range(n - 1):
        top.graph.add_edge(i, i + 1)
    sim.show_progress = False
    frames = []
    sim.observe.particle_positions(stride, callback=lambda x: frames.append(
        np.array([[p[0], p[1], p[2]] for p in x])), save=False)
    sim.run(int(n_steps), dt, show_summary=False)
    if not frames:
        raise RuntimeError("no frames recorded")
    return np.array([_unwrap(f, box) for f in frames]), box


def _unwrap(pos, box):
    """Undo the periodic fold along the chain, choosing the image nearest the previous bead."""
    out = np.empty_like(pos)
    out[0] = pos[0]
    for i in range(1, len(pos)):
        d = pos[i] - out[i - 1]
        d -= box * np.round(d / box)
        out[i] = out[i - 1] + d
    return out


def observables(frames, burn_frac=0.5):
    """Time-averaged chain statistics from a trajectory of unwrapped positions."""
    f = frames[int(len(frames) * burn_frac):]
    b = np.diff(f, axis=1)
    blen = np.linalg.norm(b, axis=2)
    u = b / blen[:, :, None]
    cos = (u[:, :-1] * u[:, 1:]).sum(axis=2)
    ree = f[:, -1] - f[:, 0]
    centre = f.mean(axis=1, keepdims=True)
    rg2 = ((f - centre) ** 2).sum(axis=2).mean(axis=1)
    return {'mean_cos': float(cos.mean()), 'b_mean': float(blen.mean()),
            'b_sq': float((blen ** 2).mean()),
            'r_ee2': float((ree ** 2).sum(axis=1).mean()),
            'rg2': float(rg2.mean()), 'n_frames': len(f)}


def persistence_length(obs):
    """From the one-step cosine, which is the estimator the lattice side is compared against."""
    c = obs['mean_cos']
    return -obs['b_mean'] / np.log(c) if 0.0 < c < 1.0 else float('inf')


def _one_run(job):
    """One simulation in its own process. Top level so it can be sent to a spawned worker."""
    kind, kw = job
    frames, _ = run(**kw)
    o = observables(frames)
    o['lp_over_b'] = persistence_length(o) / o['b_mean']
    o.update({k: kw.get(k) for k in ('n', 'k_angle', 'seed')})
    return o


def parallel_map(jobs, workers=None):
    """Run independent simulations across processes.

    Each evaluation here is an independent stochastic estimate, so seeds parallelise perfectly and
    the wall time for a given error bar falls with the worker count rather than with the run length.
    That matters because the number of INDEPENDENT samples in one trajectory is the run length
    divided by the relaxation time, not the number of recorded frames: a single long run of twelve
    relaxation times carries about twelve samples however densely it is sampled, which is a thirty
    percent error on a squared end to end distance.
    """
    import multiprocessing as mp
    workers = workers or min(len(jobs), max(1, mp.cpu_count() // 4))
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers) as pool:
        return pool.map(_one_run, jobs)


def scan_persistence_length_parallel(ks, seeds=4, n=32, bond_len=BOND_EDGE, dt=3e-8,
                                     n_steps=400000, stride=2000, verbose=True):
    """Persistence length across a grid of angle stiffnesses, several seeds each, in parallel."""
    jobs = [('lp', dict(n=n, n_steps=n_steps, dt=dt, stride=stride, k_angle=float(k),
                        bond_len=bond_len, seed=100 * s + 7))
            for k in ks for s in range(seeds)]
    res = parallel_map(jobs)
    rows = []
    for k in ks:
        sel = [r for r in res if abs(r['k_angle'] - k) < 1e-12]
        v = np.array([r['lp_over_b'] for r in sel], float)
        rows.append({'k_angle': float(k), 'lp_over_b': float(v.mean()),
                     'err': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                     'seeds': len(sel)})
        if verbose:
            print(f"  k_angle {k:8.3f}  Lp/<b> {rows[-1]['lp_over_b']:8.4f} "
                  f"+- {rows[-1]['err']:.4f}  ({len(sel)} seeds)")
    return rows


def _interp_log(rows, target):
    """Invert the measured relation by monotone interpolation in the logarithms.

    A global polynomial fit was tried first and was biased by about eight percent at the stiff end,
    because the relation has curvature a quadratic cannot follow over two decades. Interpolating
    between adjacent measured points uses only the local behaviour and cannot import curvature from
    the far end of the range.
    """
    x = np.log([r['k_angle'] for r in rows])
    y = np.log([r['lp_over_b'] for r in rows])
    o = np.argsort(y)
    return float(np.exp(np.interp(np.log(target), y[o], x[o])))


def match_persistence_length_parallel(target_lp_over_b, ks=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                                      seeds=4, refine_seeds=8, n=32, bond_len=BOND_EDGE, dt=3e-8,
                                      n_steps=400000, stride=2000, verbose=True):
    """Two-stage match: a coarse scan, then a local refinement around the first estimate."""
    coarse = scan_persistence_length_parallel(ks, seeds, n, bond_len, dt, n_steps, stride, verbose)
    lo_v = min(r['lp_over_b'] for r in coarse)
    hi_v = max(r['lp_over_b'] for r in coarse)
    if not (lo_v <= target_lp_over_b <= hi_v):
        return {'ok': False, 'reason': 'target outside the scanned range', 'range': (lo_v, hi_v)}
    k0 = _interp_log(coarse, target_lp_over_b)
    fine_ks = [k0 * f for f in (0.6, 0.8, 1.0, 1.25, 1.6)]
    if verbose:
        print(f"  coarse estimate k_angle {k0:.4f}, refining locally")
    fine = scan_persistence_length_parallel(fine_ks, refine_seeds, n, bond_len, dt,
                                            n_steps, stride, verbose)
    k1 = _interp_log(fine, target_lp_over_b)
    conf = parallel_map([('lp', dict(n=n, n_steps=n_steps, dt=dt, stride=stride, k_angle=k1,
                                     bond_len=bond_len, seed=5000 + 31 * s))
                         for s in range(refine_seeds)])
    v = np.array([r['lp_over_b'] for r in conf], float)
    got, err = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    if verbose:
        print(f"  k_angle {k1:.4f} gives Lp/<b> {got:.4f} +- {err:.4f} against target "
              f"{target_lp_over_b:.4f}, off {100 * (got / target_lp_over_b - 1):+.2f}% "
              f"({abs(got - target_lp_over_b) / max(err, 1e-12):.1f} sigma)")
    return {'ok': True, 'k_angle': k1, 'lp_over_b': got, 'err': err,
            'target': target_lp_over_b, 'rel_error': got / target_lp_over_b - 1.0,
            'sigma': float(abs(got - target_lp_over_b) / max(err, 1e-12)),
            'coarse': coarse, 'fine': fine}


def scan_persistence_length(ks, n=32, bond_len=BOND_EDGE, dt=3e-8, n_steps=400000,
                            stride=2000, seed=0, verbose=True):
    """Measure the persistence length across a grid of angle force constants."""
    rows = []
    for k in ks:
        frames, _ = run(n, n_steps, dt, stride, k_angle=k, bond_len=bond_len, seed=seed)
        o = observables(frames)
        lp = persistence_length(o)
        rows.append({'k_angle': float(k), 'lp': lp, 'lp_over_b': lp / o['b_mean'],
                     'mean_cos': o['mean_cos'], 'b_mean': o['b_mean'],
                     'polydispersity': o['b_sq'] / o['b_mean'] ** 2})
        if verbose:
            print(f"  k_angle {k:8.3f}  <cos> {o['mean_cos']:.4f}  Lp {lp:8.4f}  "
                  f"Lp/<b> {lp / o['b_mean']:8.4f}")
    return rows


def match_persistence_length(target_lp_over_b, ks=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                             n=32, bond_len=BOND_EDGE, dt=3e-8, n_steps=400000, stride=2000,
                             seed=0, verbose=True):
    """Angle force constant reproducing a target persistence length, in bond units.

    Done by scanning a grid and inverting a smooth fit rather than by bisection. Each evaluation
    here is a stochastic simulation, and a root finder that steps on the difference between two
    noisy evaluations chases its own noise: it converges on whichever pair of draws happened to
    straddle the target and reports a precision it does not have. A fit over the whole grid uses
    every run, and its residual is a visible estimate of how well the relation is determined.

    The relation is smooth but not a pure power law, so the fit is quadratic in the logarithms. The
    result is confirmed by one independent run at the fitted value with a different seed, and that
    confirmation is the number reported.
    """
    rows = scan_persistence_length(ks, n, bond_len, dt, n_steps, stride, seed, verbose)
    x = np.log(np.array([r['k_angle'] for r in rows]))
    y = np.log(np.array([r['lp_over_b'] for r in rows]))
    coef = np.polyfit(x, y, 2)
    resid = float(np.std(y - np.polyval(coef, x)))
    lo, hi = float(x.min()), float(x.max())
    ty = float(np.log(target_lp_over_b))
    if not (np.polyval(coef, lo) <= ty <= np.polyval(coef, hi)):
        return {'ok': False, 'reason': 'target outside the scanned range',
                'range': (float(np.exp(np.polyval(coef, lo))),
                          float(np.exp(np.polyval(coef, hi)))), 'rows': rows}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.polyval(coef, mid) < ty:
            lo = mid
        else:
            hi = mid
    k_fit = float(np.exp(0.5 * (lo + hi)))
    frames, _ = run(n, n_steps, dt, stride, k_angle=k_fit, bond_len=bond_len, seed=seed + 977)
    o = observables(frames)
    got = persistence_length(o) / o['b_mean']
    if verbose:
        print(f"  fit residual in log persistence length {resid:.4f}")
        print(f"  k_angle {k_fit:.4f} gives Lp/<b> {got:.4f} against target "
              f"{target_lp_over_b:.4f}, off {100 * (got / target_lp_over_b - 1):+.2f}%")
    return {'ok': True, 'k_angle': k_fit, 'lp_over_b': got, 'target': target_lp_over_b,
            'rel_error': got / target_lp_over_b - 1.0, 'fit_residual': resid, 'rows': rows,
            'polydispersity': o['b_sq'] / o['b_mean'] ** 2}


def rouse_time(n, bond_len=BOND_EDGE, d1=D1_VOXEL2_PER_S):
    """Order of magnitude of the longest relaxation time of a free draining chain.

    Run length has to be set against this and not against a fixed step count. A chain of sixty four
    beads relaxes roughly sixteen times slower than one of sixteen, so a step count that equilibrates
    the short chain leaves the long one barely started, and the sweep then reports a length dependent
    bias as if it were a scaling exponent.
    """
    return (n * bond_len) ** 2 / (3.0 * np.pi ** 2 * d1)


def steps_for(n, dt, n_rouse=12.0, bond_len=BOND_EDGE, d1=D1_VOXEL2_PER_S):
    """Step count giving the requested number of Rouse times at this timestep."""
    return int(np.ceil(n_rouse * rouse_time(n, bond_len, d1) / dt))


def exponent_sweep(ns, k_angle, dt=1e-7, k_bond=50.0, n_rouse=12.0, stride_frames=400,
                   seed=0, verbose=True):
    """Mean squared end to end distance against bead count, with run length scaled by the Rouse time.

    A softer bond is used here than in the stiffness scan, which allows a larger timestep at the same
    integration accuracy, because the bond is the stiffest force in the system and therefore the one
    that sets the step. The cost is a slightly wider bond length distribution, which is reported so
    it can be compared against the lattice value rather than assumed negligible.
    """
    rows = []
    for n in ns:
        n_steps = steps_for(n, dt, n_rouse)
        stride = max(1, n_steps // stride_frames)
        frames, _ = run(n, n_steps, dt, stride, k_angle=k_angle, k_bond=k_bond, seed=seed)
        o = observables(frames)
        rows.append({'n': int(n), 'n_steps': n_steps, 'r_ee2': o['r_ee2'], 'rg2': o['rg2'],
                     'b_mean': o['b_mean'], 'polydispersity': o['b_sq'] / o['b_mean'] ** 2,
                     'mean_cos': o['mean_cos'], 'lp_over_b': persistence_length(o) / o['b_mean'],
                     'rouse_time_s': rouse_time(n), 'sim_time_s': n_steps * dt})
        if verbose:
            r = rows[-1]
            print(f"  N {n:4d}  steps {n_steps:9d}  simulated {r['sim_time_s']:.4f} s "
                  f"= {r['sim_time_s'] / r['rouse_time_s']:5.1f} Rouse times  "
                  f"<R^2> {r['r_ee2']:9.3f}  Lp/<b> {r['lp_over_b']:6.3f}  "
                  f"polydisp {r['polydispersity']:.4f}")
    m = np.log([r['n'] - 1 for r in rows])
    y = np.log([r['r_ee2'] for r in rows])
    slope = float(np.polyfit(m, y, 1)[0])
    if verbose:
        print(f"  fitted exponent {slope:.3f}   self avoiding reference 1.176   ideal 1.000")
    return {'rows': rows, 'exponent': slope}


def exponent_sweep_parallel(ns, k_angle, seeds=32, dt=1e-7, k_bond=50.0, n_rouse=12.0,
                            stride_frames=200, verbose=True):
    """Swelling exponent with enough independent samples to mean anything.

    The number of independent samples of a squared end to end distance in ONE trajectory is its
    length divided by the longest relaxation time, not the number of frames recorded. A single run
    of twelve relaxation times therefore carries about twelve samples and a thirty percent error,
    which is larger than the entire difference between the ideal and self-avoiding exponents over
    this range of chain lengths. A first attempt at this measurement returned an exponent of 0.63,
    below both, which is not a physical value; it was the error bar.

    Independent seeds fix it without lengthening any run, and they parallelise perfectly.
    """
    jobs = []
    for n in ns:
        n_steps = steps_for(n, dt, n_rouse)
        stride = max(1, n_steps // stride_frames)
        for s in range(seeds):
            jobs.append(('r2', dict(n=int(n), n_steps=n_steps, dt=dt, stride=stride,
                                    k_angle=k_angle, k_bond=k_bond, seed=1000 * s + 3)))
    res = parallel_map(jobs)
    rows = []
    for n in ns:
        sel = [r for r in res if r['n'] == n]
        v = np.array([r['r_ee2'] for r in sel], float)
        lp = np.array([r['lp_over_b'] for r in sel], float)
        rows.append({'n': int(n), 'r_ee2': float(v.mean()),
                     'err': float(v.std(ddof=1) / np.sqrt(len(v))),
                     'lp_over_b': float(lp.mean()), 'seeds': len(sel),
                     'samples': len(sel) * n_rouse})
        if verbose:
            r = rows[-1]
            print(f"  N {n:4d}  <R^2> {r['r_ee2']:9.3f} +- {r['err']:6.3f} "
                  f"({100 * r['err'] / r['r_ee2']:4.1f}%)  Lp/<b> {r['lp_over_b']:6.3f}  "
                  f"{r['seeds']} seeds, about {r['samples']:.0f} independent samples")
    x = np.log([r['n'] - 1 for r in rows])
    y = np.log([r['r_ee2'] for r in rows])
    w = 1.0 / np.array([r['err'] / r['r_ee2'] for r in rows])
    slope, icept = np.polyfit(x, y, 1, w=w)
    resid = float(np.std(y - (slope * x + icept)))
    if verbose:
        print(f"  fitted exponent {slope:.3f}, fit residual {resid:.4f}")
        print(f"  ideal chain 1.000, self avoiding 1.176")
    return {'rows': rows, 'exponent': float(slope), 'residual': resid}


def _dyn_run(job):
    """One reference trajectory reduced to its mode and displacement ladders. Top level for pickling."""
    kw = job
    p_max = kw.pop('p_max')
    frames, _ = run(**kw)
    n = frames.shape[1]
    idx = np.arange(n)
    md = np.empty((len(frames), p_max + 1, 3))
    for p in range(p_max + 1):
        w = np.cos(p * np.pi * (idx + 0.5) / n)
        md[:, p] = (w[None, :, None] * frames).sum(axis=1) / n
    com = frames.mean(axis=1)
    rel = frames - com[:, None, :]
    mid = n // 2
    max_lag = len(frames) // 4
    acf = np.empty((p_max, max_lag + 1))
    for p in range(1, p_max + 1):
        x = md[:, p] - md[:, p].mean(axis=0)
        den = float((x * x).sum(axis=1).mean())
        for k in range(max_lag + 1):
            acf[p - 1, k] = float((x[:len(x) - k] * x[k:]).sum(axis=1).mean()) / den
    g1 = np.empty(max_lag + 1); g2 = np.empty(max_lag + 1); g3 = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        g1[k] = float(((frames[k:, mid] - frames[:len(frames) - k, mid]) ** 2).sum(axis=1).mean())
        g2[k] = float(((rel[k:, mid] - rel[:len(rel) - k, mid]) ** 2).sum(axis=1).mean())
        g3[k] = float(((com[k:] - com[:len(com) - k]) ** 2).sum(axis=1).mean())
    return {'n': n, 'mode_amp': (md ** 2).sum(axis=2).mean(axis=0), 'acf': acf,
            'g1': g1, 'g2': g2, 'g3': g3, 'dt_frame': kw['dt'] * kw['stride']}


def dynamics_ladder(ns, k_angle, seeds=4, dt=1e-7, k_bond=200.0, n_rouse=100.0,
                    frames_per_tau=40.0, p_max=4, verbose=True):
    """Mode and displacement ladders on the reference, in seconds.

    Run length is set from the Rouse time and the frame cadence from the same estimate, so every
    chain length is measured over the same number of its OWN relaxation times and resolves its modes
    equally well. A fixed step count would leave the long chains covering a fraction of one
    relaxation time, which is the failure that made the lattice sweep report the run length as
    physics.
    """
    jobs = []
    for n in ns:
        tau = rouse_time(n, BOND_EDGE)
        n_steps = int(np.ceil(n_rouse * tau / dt))
        stride = max(1, int(round(tau / (frames_per_tau * dt))))
        for s in range(seeds):
            jobs.append(dict(n=int(n), n_steps=n_steps, dt=dt, stride=stride, k_angle=k_angle,
                             k_bond=k_bond, seed=2000 * s + 17, p_max=p_max))
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(len(jobs), max(1, mp.cpu_count() // 4))) as pool:
        res = pool.map(_dyn_run, jobs)
    out = []
    for n in ns:
        sel = [r for r in res if r['n'] == n]
        rec = {'n': int(n), 'seeds': len(sel), 'dt_frame': sel[0]['dt_frame'],
               'mode_amp': np.mean([r['mode_amp'] for r in sel], axis=0),
               'acf': np.mean([r['acf'] for r in sel], axis=0),
               'g1': np.mean([r['g1'] for r in sel], axis=0),
               'g2': np.mean([r['g2'] for r in sel], axis=0),
               'g3': np.mean([r['g3'] for r in sel], axis=0)}
        taus = []
        for p in range(rec['acf'].shape[0]):
            a = rec['acf'][p]
            good = np.flatnonzero(np.isfinite(a) & (a > 1.0 / np.e))
            if len(good) >= 3:
                k = np.arange(good[0], good[-1] + 1)
                sl = np.polyfit(k * rec['dt_frame'], np.log(a[k]), 1)[0]
                taus.append(-1.0 / sl if sl < 0 else np.nan)
            else:
                taus.append(np.nan)
        rec['tau_p'] = np.array(taus)
        rec['tau_1'] = float(taus[0])
        rec['ratio_1_2'] = float(taus[0] / taus[1]) if len(taus) > 1 and taus[1] > 0 else np.nan
        out.append(rec)
        if verbose:
            print(f"  N {n:4d}  tau_1 {rec['tau_1']:.4e} s   tau_1/tau_2 {rec['ratio_1_2']:5.2f}"
                  f"   ({len(sel)} seeds)")
    return out


def check_timestep(n=16, k_angle=1.0, bond_len=BOND_EDGE, k_bond=200.0, dts=(1e-6, 3e-7, 1e-7, 3e-8),
                   n_steps=200000, stride=1000, seed=1):
    """The integrator timestep is small enough when the bond length distribution stops moving.

    A harmonic bond of stiffness k at temperature one has mean squared extension about the rest
    length of one over k in each of three directions, so the measured spread is a direct read on
    whether the integrator is resolving the stiffest force in the system.
    """
    rows = []
    for dt in dts:
        frames, _ = run(n, n_steps, dt, stride, k_angle=k_angle, bond_len=bond_len,
                        k_bond=k_bond, seed=seed)
        o = observables(frames)
        var = o['b_sq'] - o['b_mean'] ** 2
        rows.append({'dt': dt, 'b_mean': o['b_mean'], 'b_var': var,
                     'b_var_harmonic': 1.0 / k_bond, 'mean_cos': o['mean_cos']})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--timestep', action='store_true')
    ap.add_argument('--scan', action='store_true', help='persistence length against angle stiffness')
    ap.add_argument('--exponent', action='store_true', help='swelling exponent against bead count')
    ap.add_argument('--k-angle', type=float, default=0.25)
    ap.add_argument('--ns', default='8,12,16,24,32')
    ap.add_argument('--seeds', type=int, default=32)
    ap.add_argument('--match', type=float, default=None,
                    help='target persistence length in BOND units')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--dt', type=float, default=1e-7)
    ap.add_argument('--steps', type=int, default=400000)
    ap.add_argument('--stride', type=int, default=2000)
    a = ap.parse_args()

    if a.timestep:
        print("timestep convergence: bond length variance against the harmonic prediction")
        print(f"  {'dt':>10} {'<b>':>9} {'var(b)':>10} {'1/k_bond':>10} {'ratio':>7} {'<cos>':>8}")
        for r in check_timestep(n=a.n):
            print(f"  {r['dt']:10.1e} {r['b_mean']:9.4f} {r['b_var']:10.5f} "
                  f"{r['b_var_harmonic']:10.5f} {r['b_var'] / r['b_var_harmonic']:7.3f} "
                  f"{r['mean_cos']:8.4f}")

    if a.scan:
        print("persistence length against the angle force constant")
        print(f"  {'k_angle':>9} {'<cos>':>8} {'<b>':>8} {'Lp voxels':>10} {'Lp/<b>':>8}")
        for k in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            frames, _ = run(a.n, a.steps, a.dt, a.stride, k_angle=k)
            o = observables(frames)
            print(f"  {k:9.2f} {o['mean_cos']:8.4f} {o['b_mean']:8.4f} "
                  f"{persistence_length(o):10.4f} {persistence_length(o) / o['b_mean']:8.4f}")

    if a.exponent:
        print(f"swelling exponent at angle force constant {a.k_angle}, run length scaled by the "
              f"Rouse time, {a.seeds} independent seeds per length")
        exponent_sweep_parallel([int(x) for x in a.ns.split(',')], a.k_angle, seeds=a.seeds,
                                dt=a.dt)

    if a.match is not None:
        print(f"matching the angle force constant to Lp/<b> = {a.match}")
        r = match_persistence_length(a.match, n=a.n, dt=a.dt, n_steps=a.steps, stride=a.stride)
        print({k: v for k, v in r.items() if k != 'rows'})


if __name__ == '__main__':
    main()
