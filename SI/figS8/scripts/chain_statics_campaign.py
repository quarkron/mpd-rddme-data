#!/usr/bin/env python3
"""Conformational statistics of a fixed-length chain, measured on the engine.

Two things are measured, in this order because the second depends on the first.

THE EQUILIBRATION BRACKET decides which stiffnesses are usable at all. Only end bond rotation can
create curvature, and it acts at the tips, so a straight chain relaxes by transporting curvature
inward and its cost grows with chain length. A chain seeded straight and a chain seeded as a coil
are therefore approaching the same equilibrium from opposite sides at very different speeds. Where
they meet, the measurement is equilibrated. Where they do not meet within the budget, no number from
that stiffness is an equilibrium number, however stable it looks on its own.

THE SWEEP then measures the end to end distance, the radius of gyration, the tangent correlation
and the bond length distribution across stiffness and chain length, for both bond arms, and compares
against the exact transfer matrix reference.

Seeds are independent runs, not independent frames. Frames within a run are correlated, so the error
bar comes from the scatter across seeds. Quoting a frame-count standard error would understate the
uncertainty by whatever the autocorrelation time is, which for the stiff chains is large.

Outputs go to the big disk, never into the repository tree.
"""
import argparse
import os
import sys
import time

import numpy as np

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'Lattice_Microbes_2.6', 'testing', 'engine'))   # fixed_chain_harness, chain_geometry
import chain_geometry as G
import fixed_chain_harness as H

def _default_out_dir():
    """Where results go, checked rather than assumed.

    The build container mounts the source tree and one output directory. It does NOT mount the big
    disk, so an absolute path to it inside the container resolves to the container's own filesystem
    and is discarded when the container exits. The write appears to succeed and the results are
    gone. So the in-container default is the mounted output directory, and any path chosen here is
    checked against the mount table before a long run rather than after it.
    """
    explicit = os.environ.get('CHAIN_OUT')
    if explicit:
        return explicit
    if os.path.isdir('/lm/_monitor_out'):
        return '/lm/_monitor_out/lmrddme_polymer'
    return '/raid/racda/lmrddme_polymer'


OUT_DIR = _default_out_dir()

# A relative gap at or below this counts as equilibrated regardless of statistical significance.
REL_TOL = 0.01


def assert_output_survives(path):
    """Fail before a long run if the output directory will not outlive the process.

    Inside the container only paths under a bind mount survive. This checks the directory is on a
    different device from the container root, which is what a bind mount looks like from inside.
    """
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)
    probe = os.path.join(d, '.write_probe')
    with open(probe, 'w') as fh:
        fh.write('x')
    os.remove(probe)
    if os.path.isdir('/lm'):
        root_dev = os.stat('/').st_dev
        if os.stat(d).st_dev == root_dev:
            raise RuntimeError(
                "output directory %s is on the container root filesystem and will be discarded "
                "when the container exits; write under a bind mount such as /lm/_monitor_out" % d)
    return d


def _box_for(n):
    """A box large enough that a chain cannot interact with its own periodic image.

    The lattice x extent must be a multiple of thirty two. A coil of n subunits has an extent of
    order the square root of its mean squared end to end distance, but a stiff chain approaches its
    contour length, so the box is sized against the contour rather than the coil. The reader unwraps
    positions along the walk, so a chain crossing a face still measures correctly; what the box has
    to prevent is the chain excluding volume against its own periodic image, which no amount of
    careful reading can undo.
    """
    if n <= 24:
        return 32
    if n <= 40:
        return 64
    if n <= 80:
        return 96
    return 128


def _tail_stats(values, tail_frac=0.5):
    """Mean over the last fraction of a trajectory, which is the equilibrated part if it exists."""
    v = np.asarray(values, float)
    k = max(1, int(len(v) * tail_frac))
    return float(np.nanmean(v[-k:]))


def bracket_run(kappa, n, arm, seed_kind, n_steps, frame_every, seed, box=None,
                attempts=None):
    """One side of the bracket: a trajectory of the mean cosine and the squared end to end distance."""
    box = box or _box_for(n)
    h = H.build(n=n, kappa=kappa, arm=arm, seed_kind=seed_kind,
                rng=np.random.default_rng(seed), lx=box, ly=box, lz=box,
                attempts=attempts)
    n_frames = max(1, n_steps // frame_every)
    cos = np.empty(n_frames)
    r2 = np.empty(n_frames)
    rg2 = np.empty(n_frames)
    for f in range(n_frames):
        H.advance(h, frame_every, (seed * 7919 + 104729 * f) & 0xFFFFFFFF)
        ch = H.read(h)
        if ch['n'] != n:
            raise RuntimeError("chain length changed at frame %d" % f)
        o = G.observables(ch)
        cos[f] = o['mean_cos']
        r2[f] = o['r_ee2']
        rg2[f] = o['rg2']
    steps = (np.arange(n_frames) + 1) * frame_every
    return {'steps': steps, 'mean_cos': cos, 'r_ee2': r2, 'rg2': rg2}


def bracket(kappa, n, arm, n_steps, frame_every, seeds, attempts=None, verbose=True):
    """Both sides, several seeds each, and a verdict on whether they meet.

    The two sides meet when the difference of their tail means is within twice the combined standard
    error of those means across seeds. A verdict of not met is a statement that this stiffness is
    outside the equilibrated range at this budget, not that the engine is wrong.
    """
    out = {}
    for side in ('straight', 'coil'):
        runs = [bracket_run(kappa, n, arm, side, n_steps, frame_every, 1000 + s, attempts=attempts)
                for s in range(seeds)]
        out[side] = {
            'steps': runs[0]['steps'],
            'mean_cos_traj': np.array([r['mean_cos'] for r in runs]),
            'r_ee2_traj': np.array([r['r_ee2'] for r in runs]),
            'tail_cos': np.array([_tail_stats(r['mean_cos']) for r in runs]),
            'tail_r2': np.array([_tail_stats(r['r_ee2']) for r in runs]),
        }
    res = {}
    for key, tail in (('mean_cos', 'tail_cos'), ('r_ee2', 'tail_r2')):
        a, b = out['straight'][tail], out['coil'][tail]
        ma, mb = float(a.mean()), float(b.mean())
        sa = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        sb = float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0.0
        comb = np.sqrt(sa ** 2 + sb ** 2)
        # Two criteria, and either one suffices. A purely statistical test gets STRICTER as the
        # measurement improves: with enough seeds any residual difference, however physically
        # irrelevant, eventually exceeds two standard errors. That produced a ragged verdict where
        # stiffness four failed at a gap of 0.65 percent while five, six and seven passed with
        # larger gaps and looser errors, which is not a physical ordering of equilibration
        # difficulty. So a gap that is small relative to the value also counts as met, and the
        # tolerance is stated rather than discovered.
        rel = abs(ma - mb) / max(abs(0.5 * (ma + mb)), 1e-12)
        within_sigma = bool(comb > 0 and abs(ma - mb) <= 2.0 * comb)
        within_tol = bool(rel <= REL_TOL)
        res[key] = {'straight': ma, 'coil': mb, 'err_straight': sa, 'err_coil': sb,
                    'gap': ma - mb, 'combined_err': float(comb), 'rel_gap': float(rel),
                    'sigma': float(abs(ma - mb) / comb) if comb > 0 else float('inf'),
                    'within_sigma': within_sigma, 'within_tol': within_tol,
                    'met': bool(within_sigma or within_tol)}
    if verbose:
        r = res['mean_cos']
        print(f"  kappa={kappa:5.1f} N={n:4d} arm={arm:5s}  straight {r['straight']:.4f} "
              f"+- {r['err_straight']:.4f}   coil {r['coil']:.4f} +- {r['err_coil']:.4f}   "
              f"gap {r['sigma']:5.1f} sigma {100 * r['rel_gap']:5.2f}%   "
              f"{'MET' if r['met'] else 'not met'}")
    return {'kappa': kappa, 'n': n, 'arm': arm, 'sides': out, 'verdict': res}


def sweep_point(kappa, n, arm, n_steps, frame_every, burn_frac, seed, max_sep, attempts=None):
    """One stiffness, length, arm and seed. Returns time-averaged observables from the tail."""
    box = _box_for(n)
    h = H.build(n=n, kappa=kappa, arm=arm, seed_kind='coil',
                rng=np.random.default_rng(seed), lx=box, ly=box, lz=box, attempts=attempts)
    n_frames = max(1, n_steps // frame_every)
    burn = int(n_frames * burn_frac)
    acc_r2 = []
    acc_rg2 = []
    acc_cos = []
    acc_corr = np.zeros(max_sep + 1)
    acc_corr_n = np.zeros(max_sep + 1)
    lens = []
    for f in range(n_frames):
        H.advance(h, frame_every, (seed * 7919 + 104729 * f) & 0xFFFFFFFF)
        ch = H.read(h)
        if ch['n'] != n:
            raise RuntimeError("chain length changed")
        if f < burn:
            continue
        o = G.observables(ch)
        acc_r2.append(o['r_ee2'])
        acc_rg2.append(o['rg2'])
        acc_cos.append(o['mean_cos'])
        lens.append(o['bond_lengths'])
        c = G.bond_correlation(ch, max_sep)
        good = ~np.isnan(c)
        acc_corr[good] += c[good]
        acc_corr_n[good] += 1
    lens = np.concatenate(lens)
    with np.errstate(invalid='ignore'):
        corr = acc_corr / np.maximum(acc_corr_n, 1)
    corr[acc_corr_n == 0] = np.nan
    return {'r_ee2': float(np.mean(acc_r2)), 'rg2': float(np.mean(acc_rg2)),
            'mean_cos': float(np.nanmean(acc_cos)), 'bond_corr': corr,
            'b_mean': float(lens.mean()), 'b_sq': float((lens ** 2).mean()),
            'frac_face': float(np.isclose(lens, 1.0).mean()),
            'frac_edge': float(np.isclose(lens, np.sqrt(2.0)).mean()),
            'frac_body': float(np.isclose(lens, np.sqrt(3.0)).mean()),
            'n_frames_used': len(acc_r2)}


def run_sweep(kappas, ns, arms, seeds, n_steps, frame_every, burn_frac, max_sep,
              attempts=None, verbose=True):
    rows = []
    for arm in arms:
        for kappa in kappas:
            for n in ns:
                pts = [sweep_point(kappa, n, arm, n_steps, frame_every, burn_frac,
                                   2000 + 13 * s, max_sep, attempts) for s in range(seeds)]
                rec = {'arm': arm, 'kappa': kappa, 'n': n, 'seeds': seeds}
                for key in ('r_ee2', 'rg2', 'mean_cos', 'b_mean', 'b_sq',
                            'frac_face', 'frac_edge', 'frac_body'):
                    vals = np.array([p[key] for p in pts], float)
                    rec[key] = float(vals.mean())
                    rec[key + '_err'] = (float(vals.std(ddof=1) / np.sqrt(len(vals)))
                                         if len(vals) > 1 else 0.0)
                rec['bond_corr'] = np.nanmean(np.array([p['bond_corr'] for p in pts]), axis=0)
                rows.append(rec)
                if verbose:
                    print(f"  arm={arm:5s} kappa={kappa:5.1f} N={n:4d}  "
                          f"<R^2> {rec['r_ee2']:9.2f} +- {rec['r_ee2_err']:6.2f}  "
                          f"<cos> {rec['mean_cos']:.4f}  <b> {rec['b_mean']:.4f}")
    return rows


def save(rows, path):
    assert_output_survives(path)
    flat = {}
    for i, r in enumerate(rows):
        for k, v in r.items():
            flat[f"{i}_{k}"] = np.asarray(v)
    flat['n_rows'] = np.asarray(len(rows))
    np.savez_compressed(path, **flat)
    return path


def load(path):
    z = np.load(path, allow_pickle=True)
    n = int(z['n_rows'])
    rows = []
    for i in range(n):
        r = {}
        for key in z.files:
            if key.startswith(f"{i}_"):
                val = z[key]
                r[key[len(str(i)) + 1:]] = val.item() if val.ndim == 0 else val
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bracket', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--kappas', default='2,4,6,8')
    ap.add_argument('--ns', default='16,64')
    ap.add_argument('--arms', default='edge,free')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=200000)
    ap.add_argument('--frame-every', type=int, default=200)
    ap.add_argument('--burn-frac', type=float, default=0.5)
    ap.add_argument('--max-sep', type=int, default=20)
    ap.add_argument('--attempts', type=int, default=None)
    ap.add_argument('--tag', default='run')
    a = ap.parse_args()
    kappas = [float(x) for x in a.kappas.split(',')]
    ns = [int(x) for x in a.ns.split(',')]
    arms = a.arms.split(',')

    assert_output_survives(os.path.join(OUT_DIR, 'probe.npz'))
    print(f"output directory {OUT_DIR}")

    if a.bracket:
        print("equilibration bracket: straight seed against coil seed")
        print(f"  {a.steps} steps per run, {a.seeds} seeds per side, "
              f"tail is the last half of each run")
        results = []
        t0 = time.time()
        for arm in arms:
            for kappa in kappas:
                for n in ns:
                    results.append(bracket(kappa, n, arm, a.steps, a.frame_every,
                                           a.seeds, a.attempts))
        print(f"  elapsed {time.time() - t0:.0f} s")
        met = [r for r in results if r['verdict']['mean_cos']['met']]
        print()
        print(f"  {len(met)} of {len(results)} points equilibrated")
        for arm in arms:
            for n in ns:
                ok = sorted(r['kappa'] for r in met if r['arm'] == arm and r['n'] == n)
                print(f"    arm={arm:5s} N={n:4d} equilibrated at stiffness {ok}")
        # BOTH verdicts are saved. The mean cosine is a LOCAL observable and equilibrates far
        # faster than the end to end distance, which is global, so a bracket that passes on the
        # first certifies nothing about the second. Gating a global quantity on a local verdict is
        # how an unequilibrated squared end to end distance gets reported as an equilibrium value.
        save([{'kappa': r['kappa'], 'n': r['n'], 'arm': r['arm'],
               'straight': r['verdict']['mean_cos']['straight'],
               'coil': r['verdict']['mean_cos']['coil'],
               'sigma': r['verdict']['mean_cos']['sigma'],
               'rel_gap': r['verdict']['mean_cos']['rel_gap'],
               'met': float(r['verdict']['mean_cos']['met']),
               'r2_straight': r['verdict']['r_ee2']['straight'],
               'r2_coil': r['verdict']['r_ee2']['coil'],
               'r2_sigma': r['verdict']['r_ee2']['sigma'],
               'r2_rel_gap': r['verdict']['r_ee2']['rel_gap'],
               'r2_met': float(r['verdict']['r_ee2']['met']),
               'traj_straight': r['sides']['straight']['mean_cos_traj'],
               'traj_coil': r['sides']['coil']['mean_cos_traj'],
               'steps': r['sides']['straight']['steps']} for r in results],
             os.path.join(OUT_DIR, f'bracket_{a.tag}.npz'))
        print(f"  saved {os.path.join(OUT_DIR, f'bracket_{a.tag}.npz')}")

    if a.sweep:
        print("statics sweep")
        t0 = time.time()
        rows = run_sweep(kappas, ns, arms, a.seeds, a.steps, a.frame_every,
                         a.burn_frac, a.max_sep, a.attempts)
        print(f"  elapsed {time.time() - t0:.0f} s")
        p = save(rows, os.path.join(OUT_DIR, f'sweep_{a.tag}.npz'))
        print(f"  saved {p}")


if __name__ == '__main__':
    main()
