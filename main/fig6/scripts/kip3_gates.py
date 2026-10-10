"""Score the gate runs of kip3_run.py. Host python is enough (numpy only).

    python kip3_gates.py out/gate_pass_exc.npz out/gate_pass_pt.npz out/gate_dilute.npz ...

Each file is scored by the mode recorded in its metadata:
    passing   order of the two motors in lane 0 over time; any frame with two in one voxel
    dilute    tracer speed by frame-to-frame nearest-neighbour linking (tracers are 100 sites
              apart, a frame is <= 6 sites of travel), against v0
    attach    motors in lane 0 (must stay at N with exclusion), and k_on realised in the empty
              lanes from rho(t) = 1 - exp(-k_on t), against the target
    traffic   totals, tip occupancy, reservoir depletion and the largest voxel occupancy
"""
import json, sys
import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    return meta, {k: z[k] for k in z.files if k != 'meta'}


def score_passing(meta, d):
    dens = d['dens'][:, 0, :]                     # lane 0, (F, L)
    pos = [np.repeat(np.arange(dens.shape[1]), dens[f]) for f in range(dens.shape[0])]
    two_in_one = int((dens >= 2).any(axis=1).sum())
    n = np.array([len(p) for p in pos])
    lost = int((n != 2).sum())
    # With no lateral hops and two motors, order is preserved iff the gap never closes to zero
    # and the two never share a voxel; the rear one identified by position is always the rear.
    gaps = np.array([p[1] - p[0] if len(p) == 2 else np.nan for p in pos])
    print('  passing: frames %d, two-in-one-voxel frames %d, frames with !=2 motors in lane 0 %d'
          % (len(pos), two_in_one, lost))
    print('           gap min %s  median %s  final positions %s'
          % (np.nanmin(gaps), np.nanmedian(gaps), pos[-1].tolist()))
    print('  -> %s' % ('EXCLUSION HOLDS (never shared a voxel)' if two_in_one == 0 else
                       'motors shared a voxel in %d frames' % two_in_one))


def link_speed(trac, times, h_nm, cut=10):
    """Nearest-neighbour linking of tracers within a lane between consecutive frames. Links
    that start or end within `cut` sites of the plus end are dropped: with no end-dissociation
    a tracer that reaches the tip parks there, and 3 of 10 seeded per lane do so within 60 s,
    which is a 5/6 bias on the mean if kept."""
    F, NL, L = trac.shape
    dx, dt = [], []
    for l in range(NL):
        prev = None
        for f in range(F):
            cur = np.repeat(np.arange(L), trac[f, l])
            if prev is not None and len(cur) == len(prev) and len(cur):
                # both sorted; equal counts -> same order (no passing within a lane)
                ok = (cur < L - cut) & (prev < L - cut)
                dx.extend((cur[ok] - prev[ok]).tolist())
                dt.extend([times[f] - times[f - 1]] * int(ok.sum()))
            prev = cur
    dx = np.array(dx, float); dt = np.array(dt, float)
    v = dx.sum() / dt.sum() * h_nm
    per = dx / dt * h_nm
    return v, per


def sidestep_rate(trac, times):
    """Lower bound on the lane-change rate per tracer: half the summed |change| of per-lane
    counts between frames, over tracer-seconds. Opposite changes within one frame cancel, so
    use short frames."""
    counts = trac.sum(2).astype(int)                 # (F, NL)
    ev = 0.5 * np.abs(np.diff(counts, axis=0)).sum()
    return ev / (trac[0].sum() * (times[-1] - times[0]))


def score_dilute(meta, d):
    v, per = link_speed(d['trac'], d['times'], meta['h_nm'])
    n_end = int(d['trac'][-1].sum()); n_0 = int(d['trac'][0].sum())
    print('  dilute: tracers %d -> %d, mean speed %.2f nm/s (target %.1f, %.1f %% off), '
          'per-link sd %.1f nm/s, backward links %.2e of %d'
          % (n_0, n_end, v, meta['v0_nm_s'], 100 * (v / meta['v0_nm_s'] - 1), per.std(),
             (per < 0).mean(), len(per)))
    if meta['lanes'] == 'switching':
        r = sidestep_rate(d['trac'], d['times'])
        print('          lane changes >= %.3f per tracer per s (target 2 x %.2f = %.2f; a lower '
              'bound, cancellations within a frame are missed)' % (r, 0.30, 0.60))


def score_attach(meta, d):
    dens = d['dens']; t = d['times']
    k_on = meta['rates']['k_on']
    if 'M' not in meta['seeded']:
        # no full lane (point arm): k_on from the slope of the total motor count over a run
        # short enough that the tip has not filled
        tot = d['ntot'][:, 0] + d['ntot'][:, 1]
        slope = float(np.polyfit(t, tot, 1)[0]) / (dens.shape[1] * dens.shape[2])
        print('  attach (no full lane): total motors %d -> %d, k_on from slope %.4g s^-1 vs '
              'target %.4g (%.1f %%); max motors in a voxel %d'
              % (tot[0], tot[-1], slope, k_on, 100 * (slope / k_on - 1), d['maxocc'].max()))
        return
    lane0 = dens[:, 0, :].sum(1)
    others = dens[:, 1:, :].mean(axis=(1, 2))            # occupancy per site, empty lanes
    # rho(t) = 1 - exp(-k t): fit k from the early frames where rho < 0.6
    m = (others > 0.02) & (others < 0.6) & (t > 0)
    k_fit = float(np.polyfit(t[m], -np.log(1 - others[m]), 1)[0]) if m.sum() > 3 else float('nan')
    print('  attach: lane 0 motors min/max %d/%d (target %d exactly);'
          % (lane0.min(), lane0.max(), meta['sites']))
    print('          empty lanes rho(end) %.3f, fitted k_on %.4g s^-1 vs target %.4g (%.1f %%)'
          % (others[-1], k_fit, k_on, 100 * (k_fit / k_on - 1)))
    print('          max motors in a voxel %d; reservoir C final/initial %.4f'
          % (d['maxocc'].max(), d['ntot'][-1, 2] / d['ntot'][0, 2]))


def score_traffic(meta, d):
    dens = d['dens']; t = d['times']
    rho = dens.mean(axis=(1, 2))
    tip = dens[:, :, -1].mean(1)
    print('  traffic: rho(end) %.3f  tip occupancy(end) %.2f per voxel  max motors/voxel %d'
          % (rho[-1], tip[-1], d['maxocc'].max()))
    print('           reservoir C final/initial %.4f; totals M,T,C %s'
          % (d['ntot'][-1, 2] / d['ntot'][0, 2], d['ntot'][-1].tolist()))
    prof = dens[-1].mean(0)
    q = np.quantile(prof, [0.1, 0.5, 0.9])
    print('           final profile rho quantiles 10/50/90 %% = %.2f %.2f %.2f; last 50 sites %.2f'
          % (q[0], q[1], q[2], prof[-50:].mean()))
    # Jam length: sites with lane-averaged occupancy above 1/2 (Leduc's HD/LD coexistence has
    # rho_LD ~ 0.1 and rho_HD ~ 0.9 at the Fig. 1C point, so the midpoint is unambiguous),
    # smoothed over 20 sites so a single dense voxel does not count as a jam.
    lane_avg = dens.mean(1)                                   # (F, L)
    k = np.ones(20) / 20.0
    sm = np.array([np.convolve(row, k, mode='same') for row in lane_avg])
    L_jam = (sm > 0.5).sum(1)
    marks = [np.argmin(np.abs(t - tt)) for tt in (50, 100, 200, 300, 400, 600) if tt <= t[-1]]
    print('           jam length (sites, rho>1/2): ' + '  '.join(
        't=%.0f:%d' % (t[i], L_jam[i]) for i in marks))
    print('           mean rho in the last 100 sites %.2f, first 500 sites %.2f'
          % (lane_avg[-1, -100:].mean(), lane_avg[-1, :500].mean()))


def main():
    for path in sys.argv[1:]:
        meta, d = load(path)
        print('%s  [%s / %s / %s]  wall %.1f s' % (path, meta['mode'], meta['arm'], meta['lanes'],
                                                   meta['wall_s']))
        dict(passing=score_passing, dilute=score_dilute, attach=score_attach,
             traffic=score_traffic)[meta['mode']](meta, d)


if __name__ == '__main__':
    main()
