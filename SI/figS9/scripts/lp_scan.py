"""Converged persistence-length scan of the off-lattice (ReaDDy) chain against its angle stiffness (2026-10-09).

    /raid/racda/opt/readdy_env/bin/python3 lp_scan.py [--n 11] [--steps 3200000] [--seeds 8] [--workers 96] [--out ../data/lp_scan_n11.npz]

The chain of fig:polymer c's off-lattice arms: n beads, bond sqrt(2) voxels, harmonic angle k_angle (theta - pi)^2,
bead D 0.3 um^2/s, dt 3e-8 s (readdy_chain.py, unchanged). L_p/<b> from the one-step bond cosine, the estimator the
lattice side uses, over the second half of each run. Every (k_angle, seed) runs from BOTH a coil and a straight start:
the coil approaches equilibrium from below in stiffness and the straight start from above, so their agreement is the
evidence of convergence the Sep runs (4e5 steps, coil only) did not have. The grid includes 2.3054, the k_angle the
fig:polymer c ReaDDy arms (readdy_crowd_campaign.sh, readdy_chain_bath.py) actually use.
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import readdy_chain as RC                      # noqa: E402

KS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.3054, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=11); ap.add_argument('--steps', type=int, default=3200000)
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--ks', default=','.join(str(k) for k in KS)); ap.add_argument('--stride', type=int, default=4000)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    ks = [float(k) for k in A.ks.split(',')]
    jobs = [('lp', dict(n=A.n, n_steps=A.steps, dt=3e-8, stride=A.stride, k_angle=k, seed=1000 * s + 17, start=st))
            for k in ks for s in range(A.seeds) for st in ('coil', 'straight')]
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(min(A.workers, len(jobs))) as pool:
        res = pool.map(RC._one_run, jobs)
    for r, (_, kw) in zip(res, jobs):
        r['start'] = kw['start']
    print('%d runs in %.0f s' % (len(res), time.time() - t0))
    rows = []
    for k in ks:
        line = '  k_angle %6.3f' % k
        for st in ('coil', 'straight'):
            v = np.array([r['lp_over_b'] for r in res if abs(r['k_angle'] - k) < 1e-12 and r['start'] == st])
            line += '  %-8s %.4f +- %.4f' % (st, v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
        allv = np.array([r['lp_over_b'] for r in res if abs(r['k_angle'] - k) < 1e-12])
        rows.append((k, allv.mean(), allv.std(ddof=1) / np.sqrt(len(allv))))
        print(line + '   both %.4f +- %.4f' % rows[-1][1:])
    np.savez(A.out, rows=np.array(rows), runs=np.array([(r['k_angle'], 0 if r['start'] == 'coil' else 1, r['seed'], r['lp_over_b'],
             r['mean_cos'], r['b_mean']) for r in res]), n=A.n, steps=A.steps, dt=3e-8)
    print('wrote', A.out)
