#!/usr/bin/env python3
"""Conformational dynamics of a fixed-length chain on the engine.

WHAT CAN AND CANNOT BE CLAIMED. The interior move set is entirely local: end bond rotation moves one
subunit, kink moves one, and the crankshaft moves a segment of at most six. There is no pivot move.
That is what makes a dynamics question meaningful at all, because a local move set produces Rouse
dynamics and Rouse is the correct physics for a chain without hydrodynamic interactions. A sampler
with a pivot move would relax correctly and mean nothing dynamically.

THE MEASUREMENTS, in increasing order of what they assume.

The mode ladder needs no reference and no time unit. Rouse mode amplitudes should follow an inverse
squared sine in the mode index, and mode relaxation times should follow the same form, so that the
ratio of the first to the second is four with no free parameter. A sampler can only pass this by
having the right internal dynamics.

The scaling ladder needs no reference either, only several chain lengths. The longest relaxation
time should grow as a power of the chain length and the centre of mass diffusion coefficient should
fall as one over it.

THE ATTEMPT BUDGET IS THE POINT OF THE SECOND LADDER. The kernel budgets interior move attempts per
COMPLEX and not per subunit, so with a fixed budget each subunit's attempt rate falls as one over
the chain length. That multiplies every relaxation time by an extra factor of the chain length. Both
budgets are therefore measured: the as-shipped constant, and a budget scaled with the chain length.
If the exponents differ by one, that is a property of the budget and not of the move set, and the
remedy is to scale the budget.

DISPLACEMENT NEEDS CARE. Positions are absolute lattice coordinates, so a chain crossing a periodic
face jumps by a box length between frames. Every trajectory is unwrapped frame to frame against the
previous position before any displacement is taken.
"""
import argparse
import os

import numpy as np

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'Lattice_Microbes_2.6', 'testing', 'engine'))   # fixed_chain_harness, chain_geometry
import chain_geometry as G
import fixed_chain_harness as H
# The Sep 6 reorganisation left this campaign in studies/polymer and its statics sibling in studies/lmrddme_validation,
# so the import fails from anywhere. Locate the sibling rather than assume a working directory: without this the whole
# module is unimportable, and fcc_transport_analysis.py, which borrows its estimators, goes down with it.
_SIB = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'lmrddme_validation')
if _os.path.isdir(_SIB) and _SIB not in _sys.path:
    _sys.path.insert(0, _SIB)
from chain_statics_campaign import OUT_DIR, _box_for, assert_output_survives, save


def trajectory(n, kappa, arm, n_frames, steps_per_frame, seed, burn_frames=0,
               attempts=None, rigid_translation=False, box=None):
    """Bead positions at every frame, unwrapped frame to frame, plus the physical time axis."""
    box = box or _box_for(n)
    h = H.build(n=n, kappa=kappa, arm=arm, seed_kind='coil', rng=np.random.default_rng(seed),
                lx=box, ly=box, lz=box, attempts=attempts,
                rigid_translation=rigid_translation)
    pos = np.empty((n_frames, n, 3))
    prev = None
    for f in range(burn_frames + n_frames):
        H.advance(h, steps_per_frame, (seed * 7919 + 104729 * f) & 0xFFFFFFFF)
        ch = H.read(h)
        if ch['n'] != n:
            raise RuntimeError("chain length changed at frame %d" % f)
        p = ch['pos'].astype(float)
        if prev is not None:
            p = prev + (p - prev) - box * np.round((p - prev) / box)
        prev = p
        if f >= burn_frames:
            pos[f - burn_frames] = p
    dt = H.physical_time(steps_per_frame, h['tau_s'], h['sol'].getTauScale())
    return {'pos': pos, 'dt': dt, 'steps_per_frame': steps_per_frame,
            'attempts': h['attempts'], 'n': n, 'kappa': kappa, 'arm': arm, 'box': box}


def modes(pos, p_max):
    """Rouse modes of a whole trajectory. Mode zero is the centre of mass."""
    n = pos.shape[1]
    idx = np.arange(n)
    out = np.empty((pos.shape[0], p_max + 1, 3))
    for p in range(p_max + 1):
        w = np.cos(p * np.pi * (idx + 0.5) / n)
        out[:, p] = (w[None, :, None] * pos).sum(axis=1) / n
    return out


def mode_amplitudes(md):
    """Time-averaged squared mode amplitude, which should follow an inverse squared sine."""
    return (md ** 2).sum(axis=2).mean(axis=0)


def rouse_amplitude_prediction(n, p_max):
    """Ideal-chain mode amplitude shape, up to one overall scale.

    Only the SHAPE is predicted here. The scale carries the bond length and the chain stiffness, so
    comparing shapes normalised at the first mode tests the mode structure without smuggling in a
    fitted stiffness.
    """
    p = np.arange(p_max + 1, dtype=float)
    with np.errstate(divide='ignore'):
        amp = 1.0 / (4.0 * n * np.sin(p * np.pi / (2.0 * n)) ** 2)
    amp[0] = np.nan
    return amp


def autocorrelation(series, max_lag):
    """Normalised vector autocorrelation, averaged over origins."""
    x = series - series.mean(axis=0, keepdims=True)
    denom = float((x * x).sum(axis=1).mean())
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        out[k] = float((x[:len(x) - k] * x[k:]).sum(axis=1).mean()) / denom
    return out


def relaxation_time(acf, dt, floor=1.0 / np.e):
    """Relaxation time from a log-linear fit over the range where the correlation is resolved.

    Fitted only above the noise floor rather than integrated to the end. An integral of a decayed
    correlation adds a long tail of noise with equal weight to the signal, and the resulting time is
    dominated by whatever the tail happened to do.
    """
    good = np.flatnonzero(np.isfinite(acf) & (acf > floor))
    if len(good) < 3:
        return float('nan')
    k = np.arange(good[0], good[-1] + 1)
    y = np.log(acf[k])
    slope = np.polyfit(k * dt, y, 1)[0]
    return float(-1.0 / slope) if slope < 0 else float('nan')


def displacement_ladder(pos, max_lag):
    """The three displacement curves that separate internal motion from whole-chain motion.

    g1 is a subunit in the laboratory frame, g2 the same subunit relative to the centre of mass, and
    g3 the centre of mass itself. Below the longest relaxation time g1 grows as the square root of
    time for an ideal chain, and above it g1 and g3 merge. Reporting only g1 hides which of the two
    regimes a measurement is in.
    """
    com = pos.mean(axis=1)
    rel = pos - com[:, None, :]
    mid = pos.shape[1] // 2
    g1 = np.empty(max_lag + 1)
    g2 = np.empty(max_lag + 1)
    g3 = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        d1 = pos[k:, mid] - pos[:len(pos) - k, mid]
        d2 = rel[k:, mid] - rel[:len(rel) - k, mid]
        d3 = com[k:] - com[:len(com) - k]
        g1[k] = float((d1 ** 2).sum(axis=1).mean())
        g2[k] = float((d2 ** 2).sum(axis=1).mean())
        g3[k] = float((d3 ** 2).sum(axis=1).mean())
    return g1, g2, g3


def analyse(traj, p_max=8, max_lag=None):
    pos = traj['pos']
    max_lag = max_lag or len(pos) // 4
    md = modes(pos, p_max)
    amps = mode_amplitudes(md)
    acfs = np.array([autocorrelation(md[:, p], max_lag) for p in range(1, p_max + 1)])
    taus = np.array([relaxation_time(a, traj['dt']) for a in acfs])
    g1, g2, g3 = displacement_ladder(pos, max_lag)
    lags = np.arange(max_lag + 1) * traj['dt']
    # The centre of mass diffusion coefficient is the slope of g3 in its TERMINAL regime only.
    # Below the longest relaxation time the centre of mass of a chain driven by internal moves is
    # sub-diffusive, so a straight line fitted from the first lag averages two regimes and returns a
    # number that is neither. The fit therefore starts above a few relaxation times, and reports the
    # window it used so a reader can see whether the run was long enough to have one.
    # The centre of mass diffusion coefficient needs a window bounded at BOTH ends. Below a few
    # relaxation times the centre of mass is still sub-diffusive. Above the point where it has
    # wandered a sizeable fraction of the box it is confined, because the lattice has hard walls
    # rather than periodic images, and a straight line fitted through that region measures the box.
    # Measured directly: the relaxation time is unchanged across boxes of 32, 64 and 96 to within
    # seed scatter, while the diffusion coefficient from an unbounded fit moved by a factor of three
    # over the same range. So the internal dynamics is safe in a small box and the centre of mass
    # motion is not.
    tau1 = float(taus[0])
    box = float(traj.get('box', 0.0))
    cap = (0.25 * box) ** 2 if box > 0 else np.inf
    d_com, fit_lo, fit_hi = float('nan'), float('nan'), float('nan')
    if np.isfinite(tau1) and tau1 > 0:
        lo = int(np.searchsorted(lags, 3.0 * tau1))
        hi = int(np.searchsorted(g3, cap))
        hi = min(hi, max_lag + 1)
        if hi - lo >= 5:
            d_com = float(np.polyfit(lags[lo:hi], g3[lo:hi], 1)[0] / 6.0)
            fit_lo, fit_hi = float(lags[lo]), float(lags[hi - 1])
    return {'n': traj['n'], 'kappa': traj['kappa'], 'arm': traj['arm'],
            'attempts': traj['attempts'], 'dt': traj['dt'],
            'mode_amp': amps, 'mode_amp_ideal': rouse_amplitude_prediction(traj['n'], p_max),
            'mode_acf': acfs, 'tau_p': taus, 'tau_1': float(taus[0]),
            'tau_ratio_1_2': float(taus[0] / taus[1]) if taus[1] > 0 else float('nan'),
            'lags': lags, 'g1': g1, 'g2': g2, 'g3': g3, 'd_com': d_com,
            'fit_window': (fit_lo, fit_hi),
            'run_over_tau1': float(lags[-1] * 4.0 / tau1) if tau1 > 0 else float('nan')}


def pilot_tau(n, kappa, arm, attempts, seed, frames=1500, steps_per_frame=20, burn=300):
    """A short run whose only job is to estimate the longest relaxation time.

    Production run length has to be set against this. A fixed step count applied to every chain
    length leaves the long chains covering a fraction of one relaxation time, and the symptom is
    subtle: the fitted relaxation time stops growing with chain length and the mode ratio collapses
    toward one, both of which look like physics and are the run length.
    """
    tr = trajectory(n, kappa, arm, frames, steps_per_frame, seed, burn_frames=burn,
                    attempts=attempts)
    a = analyse(tr, p_max=2)
    return float(a['tau_1']) if np.isfinite(a['tau_1']) else float('nan')


def plan_run(tau1, dt_step, target_tau=60.0, frames=4000, frames_per_tau=40.0):
    """Steps per frame and frame count covering the requested number of relaxation times.

    The frame count is held fixed and the cadence is stretched instead, so the cost of reading the
    lattice back does not grow with the chain length even though the run length does.
    """
    if not np.isfinite(tau1) or tau1 <= 0:
        return 20, frames
    steps_per_frame = max(1, int(round(tau1 / (frames_per_tau * dt_step))))
    need = int(np.ceil(target_tau * tau1 / (steps_per_frame * dt_step)))
    return steps_per_frame, max(frames, need)


def budget_sweep(ns, kappa, arm, n_frames, steps_per_frame, seeds, p_max, burn_frames,
                 rigid_translation=False, verbose=True, adaptive=True, target_tau=60.0,
                 budgets=('per_complex', 'per_subunit'), box=None):
    """The same measurement under both attempt budgets, which is the point of this campaign."""
    rows = []
    for budget in budgets:
        for n in ns:
            att = 1 if budget == 'per_complex' else max(1, n - 1)
            spf, nf = steps_per_frame, n_frames
            if adaptive:
                t0 = pilot_tau(n, kappa, arm, att, 7777)
                dt_step = 50e-6
                spf, nf = plan_run(t0, dt_step, target_tau)
                nf = min(nf, 12000)
                if verbose:
                    print(f"    pilot tau_1 {t0:.3e} s -> {spf} steps per frame, {nf} frames, "
                          f"covering {nf * spf * dt_step / t0:.0f} relaxation times")
            per_seed = []
            for s in range(seeds):
                tr = trajectory(n, kappa, arm, nf, spf, 3000 + 17 * s,
                                burn_frames=max(200, nf // 8), attempts=att,
                                rigid_translation=rigid_translation, box=box)
                per_seed.append(analyse(tr, p_max))
            rec = {'budget': budget, 'n': n, 'kappa': kappa, 'arm': arm, 'attempts': att,
                   'seeds': seeds, 'dt': per_seed[0]['dt']}
            rec['run_over_tau1'] = float(np.nanmean([p['run_over_tau1'] for p in per_seed]))
            for key in ('tau_1', 'tau_ratio_1_2', 'd_com'):
                v = np.array([p[key] for p in per_seed], float)
                rec[key] = float(np.nanmean(v))
                rec[key + '_err'] = (float(np.nanstd(v, ddof=1) / np.sqrt(len(v)))
                                     if len(v) > 1 else 0.0)
            for key in ('mode_amp', 'mode_amp_ideal', 'tau_p', 'lags', 'g1', 'g2', 'g3'):
                rec[key] = np.nanmean(np.array([p[key] for p in per_seed]), axis=0)
            rec['mode_acf'] = np.nanmean(np.array([p['mode_acf'] for p in per_seed]), axis=0)
            rows.append(rec)
            if verbose:
                print(f"  budget={budget:12s} N={n:4d}  tau_1 {rec['tau_1']:.4e} s "
                      f"+- {rec['tau_1_err']:.1e}  tau_1/tau_2 {rec['tau_ratio_1_2']:5.2f} "
                      f"(Rouse 4)  D_com {rec['d_com']:.4e}  run/tau_1 {rec['run_over_tau1']:.0f}")
    for budget in budgets:
        sel = sorted([r for r in rows if r['budget'] == budget], key=lambda r: r['n'])
        if len(sel) >= 2:
            x = np.log([r['n'] for r in sel])
            t = np.array([r['tau_1'] for r in sel], float)
            d = np.array([r['d_com'] for r in sel], float)
            ok = np.isfinite(t) & (t > 0)
            if ok.sum() >= 2 and verbose:
                st = float(np.polyfit(x[ok], np.log(t[ok]), 1)[0])
                okd = np.isfinite(d) & (d > 0)
                sd = float(np.polyfit(x[okd], np.log(d[okd]), 1)[0]) if okd.sum() >= 2 else np.nan
                print(f"  {budget:12s}: tau_1 exponent {st:+.3f} (Rouse ideal 2, self avoiding "
                      f"2.18)   D_com exponent {sd:+.3f} (Rouse -1)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ns', default='8,12,16,24,32')
    ap.add_argument('--kappa', type=float, default=2.0)
    ap.add_argument('--arm', default='edge')
    ap.add_argument('--frames', type=int, default=4000)
    ap.add_argument('--steps-per-frame', type=int, default=20)
    ap.add_argument('--burn-frames', type=int, default=1000)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--pmax', type=int, default=8)
    ap.add_argument('--rigid-translation', action='store_true')
    ap.add_argument('--tag', default='dyn')
    ap.add_argument('--target-tau', type=float, default=60.0)
    ap.add_argument('--no-adaptive', action='store_true')
    ap.add_argument('--budgets', default='per_complex,per_subunit')
    ap.add_argument('--box', type=int, default=None)
    a = ap.parse_args()
    assert_output_survives(os.path.join(OUT_DIR, 'probe.npz'))
    print(f"output directory {OUT_DIR}")
    print(f"dynamics: kappa {a.kappa}, arm {a.arm}, {a.frames} frames of "
          f"{a.steps_per_frame} steps, {a.seeds} seeds")
    rows = budget_sweep([int(x) for x in a.ns.split(',')], a.kappa, a.arm, a.frames,
                        a.steps_per_frame, a.seeds, a.pmax, a.burn_frames,
                        a.rigid_translation, adaptive=not a.no_adaptive,
                        target_tau=a.target_tau, budgets=tuple(a.budgets.split(',')),
                        box=a.box)
    p = save(rows, os.path.join(OUT_DIR, f'dynamics_{a.tag}.npz'))
    print(f"  saved {p}")


if __name__ == '__main__':
    main()
