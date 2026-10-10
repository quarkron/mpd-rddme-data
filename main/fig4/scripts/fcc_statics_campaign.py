#!/usr/bin/env python3
"""Equilibrium statistics of the FCC chain: engine against the exact reference.

WHAT MAKES THIS DIFFERENT FROM chain_statics_campaign.py. That campaign seeded from a straight or a
uniform self-avoiding walk and then had to prove it had equilibrated, with a two-sided bracket that
failed above a stiffness ceiling falling with chain length. Here every run starts from an EXACT
sample of the target distribution (fcc_exact.seed), so there is nothing to equilibrate and the
question changes: does the engine STAY on the distribution it was handed? A sampler that obeys
detailed balance leaves it alone; one that does not drifts off it, and the drift is the measurement.

So each point reports three things, not one: the exact value, the engine's time average, and the
drift between the first tenth of the run and the last half. The drift is what would previously have
been hidden inside an equilibration argument.

TWO REFERENCES, both exact and both cheap. The phantom chain is closed form at every finite length;
the self-avoiding chain follows from rejection on the same draws. The engine's backbone is
self-avoiding, so the self-avoiding curve is the one it must reproduce, and the gap between the two
is the excluded volume the figure is about.

  python3 fcc_statics_campaign.py --set len   --out .../statics_len.npz     # Fig 1: <R^2> vs M
  python3 fcc_statics_campaign.py --set kappa --out .../statics_kappa.npz   # Fig 2: <R^2> vs kappa
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'Lattice_Microbes_2.6', 'testing', 'engine'))
import fcc_exact as FX                                                   # noqa: E402
import fixed_chain_harness as H                                          # noqa: E402
import chain_geometry as G                                               # noqa: E402

# The SI move set (LM-RDDME SI, sec:fcc-moveset): end-bond rotation 1 | flip 2 | sideways 8 | edge translation 16.
# The earlier set was H.MOVE_ALL = end-bond | flip | crankshaft (0x7); --move-mask 7 reproduces it.
MOVE_SI = 0x1B

# Fig 1 sweeps chain length at two stiffnesses: kappa 1 where excluded volume swells the chain by
# tens of percent, and kappa 4 where it is a few percent and the test is of the size itself.
SET_LEN = [(k, n) for k in (1.0, 4.0) for n in (8, 16, 32, 64)]
# Fig 2 sweeps stiffness at one length. kappa 0 is an exact anchor: the five terms of the mean
# cosine cancel, so r = 0 identically and the chain is an ideal random walk with <R^2> = 2M.
SET_KAPPA = [(k, 11) for k in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)]


def box_for(kappa, n, rng, slack=2.0, margin=3, floor=32):
    """A lattice that holds the ensemble, not just its typical member.

    Sized from the 99.9th percentile span so the seeder never has to reject on fit, which would
    select the compact tail, and then multiplied by `slack` so a chain that swells during the run
    still does not see its own periodic image.
    """
    span = FX.required_extent(kappa, n, rng)
    return max(floor, int(np.ceil(slack * span)) + 2 * margin + 1)


def sample_recentred(h, n_frames, steps_per_frame, seed0):
    """H.sample, with the chain translated back to the box centre after every frame.

    WHY. The structure lattice is not periodic, and the interior moves alone carry a chain across it:
    measured in the single-core port (Sep 28), a chain random-walks ~100 voxels per 2e5 attempts, and a
    chain that meets a face loses its extended conformations (they no longer fit), which lowers <R^2>
    by 0.7-1.4 % at 4-6 subunits and more for longer chains. A rigid translation leaves the shape
    distribution untouched, so re-seeding the same backbone at the centre removes the wall without
    touching the statistics; the shift has an even coordinate sum so the chain stays on its FCC class.
    """
    n0 = H.read(h)['n']
    L = np.array([h['lx'], h['ly'], h['lz']])
    frames = []
    for f in range(n_frames):
        H.advance(h, steps_per_frame, (seed0 + 7919 * f) & 0xFFFFFFFF)
        ch = H.read(h)
        if ch['n'] != n0:
            raise RuntimeError("chain length changed from %d to %d at frame %d" % (n0, ch['n'], f))
        frames.append(G.observables(ch))
        pos = np.asarray(ch['pos'], np.int64)
        shift = np.rint(L / 2.0 - pos.mean(axis=0)).astype(np.int64)
        if shift.sum() % 2:
            shift[0] += 1 if shift[0] <= 0 else -1                       # keep the parity class
        if shift.any():
            H.seed_chain(h['sol'], pos + shift, ch['dirs'], h['lx'], h['ly'], h['lz'], species=h['species'])
    return frames


def one_point(kappa, n, seeds, n_frames, steps_per_frame, rng, verbose=True, move_mask=MOVE_SI):
    box = box_for(kappa, n, rng)
    m = n - 1
    exact_phantom = FX.mean_square_r(kappa, m)
    r2s, seeded, cosines, bonds = [], [], [], []
    t0 = time.time()
    for s in range(seeds):
        sub = np.random.default_rng([int(1e6 * kappa), n, s])
        h = H.build(n, kappa, arm='edge', seed_kind='exact', rng=sub, lx=box, ly=box, lz=box, move_mask=move_mask)
        st = H.read(h)
        seeded.append(float(np.sum((st['pos'][-1] - st['pos'][0]) ** 2)))
        fr = sample_recentred(h, n_frames, steps_per_frame, seed0=1000 + s)   # list of per-frame dicts
        r2s.append(np.array([f['r_ee2'] for f in fr], float))
        cosines.append(np.array([f['mean_cos'] for f in fr], float))
        bonds.append(float(np.mean([f['contour'] / (n - 1) for f in fr])))
    r2s = np.stack(r2s)                                   # (seeds, frames)
    head = r2s[:, :max(1, n_frames // 10)].mean()
    tail = r2s[:, n_frames // 2:].mean()
    per_seed = r2s.mean(axis=1)
    rec = {'kappa': kappa, 'n': n, 'box': box, 'seeds': seeds, 'move_mask': int(move_mask), 'recentred': 1,
           'r2_engine': float(per_seed.mean()),
           'r2_err': float(per_seed.std(ddof=1) / np.sqrt(seeds)),
           'r2_exact_phantom': float(exact_phantom),
           'r2_seeded': float(np.mean(seeded)),
           'drift': float(tail / head - 1.0),
           'mean_cos': float(np.mean([c.mean() for c in cosines])),
           'b_mean': float(np.mean(bonds)),
           'wall_s': time.time() - t0}
    if verbose:
        print("  kappa=%4.1f n=%3d box=%3d  <R2> engine %9.2f +- %6.2f   phantom %9.2f   "
              "seeded %9.2f   drift %+7.4f   %5.1fs"
              % (kappa, n, box, rec['r2_engine'], rec['r2_err'], exact_phantom,
                 rec['r2_seeded'], rec['drift'], rec['wall_s']), flush=True)
    return rec


def exact_saw(kappa, n, draws, rng):
    """The self-avoiding reference: rejection on exactly sampled phantom draws, so no reweighting. The estimate
    itself is Monte Carlo -- the self-avoiding chain has no closed form -- and carries a sampling error."""
    r2, _, keep = FX.sample(max(kappa, 1e-12), n - 1, draws, rng)
    if keep.sum() < 200:
        raise RuntimeError("only %d self-avoiding draws of %d at kappa %g n %d; this corner needs a "
                           "Rosenbluth grower" % (int(keep.sum()), draws, kappa, n))
    return float(r2[keep].mean()), float(r2[keep].std(ddof=1) / np.sqrt(keep.sum())), float(keep.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--set', choices=('len', 'kappa'), required=True)
    ap.add_argument('--seeds', type=int, default=32)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--steps-per-frame', dest='spf', type=int, default=200)
    ap.add_argument('--saw-draws', dest='saw_draws', type=int, default=400000)
    ap.add_argument('--only', default='', help='comma-separated kappa:n points to run, e.g. 4:32,5:11')
    ap.add_argument('--move-mask', dest='move_mask', type=lambda x: int(x, 0), default=MOVE_SI,
                    help='interior move mask (default 0x1B, the SI set; 7 = the earlier set with the crankshaft)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    points = SET_LEN if a.set == 'len' else SET_KAPPA
    if a.only:
        want = {(float(x.split(':')[0]), int(x.split(':')[1])) for x in a.only.split(',')}
        points = [p for p in points if p in want]
        if len(points) != len(want):
            raise SystemExit("--only named %s but the set holds %s" % (sorted(want), sorted(points)))
    rng = np.random.default_rng(20260909)
    print("fcc statics campaign, set=%s, %d points, %d seeds each, %d x %d interior steps"
          % (a.set, len(points), a.seeds, a.frames, a.spf), flush=True)
    recs = []
    for kappa, n in points:
        rec = one_point(kappa, n, a.seeds, a.frames, a.spf, rng, move_mask=a.move_mask)
        saw, saw_err, acc = exact_saw(kappa, n, a.saw_draws, rng)
        rec.update({'r2_exact_saw': saw, 'r2_exact_saw_err': saw_err, 'saw_accept': acc})
        print("        self-avoiding, sampled %9.2f +- %5.2f (acceptance %.3f)   engine/SAW %.4f"
              % (saw, saw_err, acc, rec['r2_engine'] / saw), flush=True)
        recs.append(rec)
    keys = sorted(recs[0])
    np.savez_compressed(a.out, n_rows=len(recs),
                        **{f"{i}_{k}": recs[i][k] for i in range(len(recs)) for k in keys})
    print("wrote %s" % a.out, flush=True)


if __name__ == '__main__':
    main()
