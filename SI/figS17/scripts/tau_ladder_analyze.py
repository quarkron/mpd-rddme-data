"""Reduce tau_ladder_rx2.sh: the operator-splitting error of the fixed and the randomized step order.

Observable: the product fraction N_C(t)/N_A(0) of the Fig. 3 reaction, the mean over 16 seeds at each frame. In the
short window the frames sit at the same physical times for every dt, so the four dt can be compared frame by frame.

  1. fixed against perm at equal dt: the pure ordering effect, which needs no exact reference;
  2. each order against its own dt -> 0 limit (a linear fit in dt through the four points), whose slope is the O(dt)
     error and whose intercepts must agree between the two orders (the same continuous dynamics);
  3. the late (long-window) composition and K_eff, which no order may move (each operator keeps the equilibrium).

  usage (lm2.6 container):  python3 tau_ladder_analyze.py
"""
import glob, json, os

import numpy as np

ROOT = '/raid/racda/campaigns/lmrddme_validation/tau_ladder'
DTS = (8e-8, 4e-8, 2e-8, 1e-8)
DTNAME = {8e-8: '8e-8', 4e-8: '4e-8', 2e-8: '2e-8', 1e-8: '1e-8'}   # as the run script spells them


def load(order, win, dt):
    fs = sorted(glob.glob(os.path.join(ROOT, '%s_randsplit' % order, win, 'rx_dt%s_s*.npz' % DTNAME[dt])))
    tau, frac, keff = None, [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        c = z['counts'].astype(float)
        tau = z['tau']
        frac.append(c[:, 2] / c[0, 0])
        keff.append(json.loads(str(z['meta']))['ratio_meas'])
    if not frac:
        return None
    n = min(len(x) for x in frac)
    a = np.array([x[:n] for x in frac])
    return tau[:n], a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a)), np.array(keff), len(a)


def main():
    for win in ('short', 'long'):
        print('== %s window: N_C/N_A(0) at selected frames, mean +- SEM over seeds' % win)
        R = {(o, dt): load(o, win, dt) for o in ('fixed', 'perm') for dt in DTS}
        if any(v is None for v in R.values()):
            print('  incomplete'); continue
        tau = R[('fixed', DTS[-1])][0]
        picks = [1, 2, 4, 8, len(tau) - 1] if win == 'short' else [1, 3, 10, 30, len(tau) - 1]
        for k in picks:
            line = '  tau %.4f |' % tau[k]
            for o in ('fixed', 'perm'):
                v = [R[(o, dt)][1][k] for dt in DTS]; e = [R[(o, dt)][2][k] for dt in DTS]
                p = np.polyfit(DTS, v, 1, w=1.0 / np.maximum(np.array(e), 1e-12))
                line += ' %s %s  ->0: %.4f (slope %+.2e/ns) |' % (o, ' '.join('%.4f' % x for x in v), p[1], p[0] * 1e-9)
            print(line)
        print('  fixed - perm at equal dt (in SEM of the difference), averaged over all frames after the first:')
        for dt in DTS:
            a, b = R[('fixed', dt)], R[('perm', dt)]
            d = (a[1] - b[1])[1:]; s = np.hypot(a[2], b[2])[1:]
            print('    dt %4.0f ns  mean diff %+.5f   mean |diff|/SEM %.2f   max |diff|/SEM %.2f'
                  % (dt * 1e9, d.mean(), np.mean(np.abs(d) / s), np.max(np.abs(d) / s)))
        if win == 'long':
            for o in ('fixed', 'perm'):
                print('  late K_eff/K_id %-5s ' % o + '  '.join('dt %2.0f ns: %.4f+-%.4f' % (dt * 1e9, R[(o, dt)][3].mean(),
                      R[(o, dt)][3].std(ddof=1) / np.sqrt(len(R[(o, dt)][3]))) for dt in DTS))


if __name__ == '__main__':
    main()
