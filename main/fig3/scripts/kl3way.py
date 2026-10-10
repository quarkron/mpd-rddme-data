"""Three-way lambda=8h comparison with ReaDDy as the reference.

KL(ReaDDy || X) over subvolume occupancy for the canonical MPD-RDDME arms and our kinetic
hard-sphere BD, plus pooled P(n) snapshots for the occupancy panels. The reference is the
pooled ReaDDy ensemble (early window, stride 25, tau <= 0.12; production window, stride
2000, tau <= 6), thinned to a log-spaced frame grid so the matched curves stay readable.
The floor is the same-distribution KL at the sample size of the noisiest arm (the 40-seed
BD fleet), measured by resampling the reference pool; the lattice arms pool 8-32x more
cells per frame, so their floor is lower still. Results cache to kl3way_cache.npz.
"""
import glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = os.path.join(HERE, 'mpd_crddme', 'rxl80c_out')
RDY = os.path.join(HERE, 'readdy_ref', 'out_field')
NMAX = 24
SNAP_TAUS = (0.03, 0.3, 3.0)
NLOG = 38                       # reference frames kept for the KL curves


def hist(counts):
    h = np.bincount(np.clip(np.asarray(counts).ravel(), 0, NMAX), minlength=NMAX + 1).astype(float)
    return h / h.sum()


def kl(p, q, eps=1e-12):
    q = np.maximum(q, eps); m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def load_all(files):
    zs = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            z['occA']               # skip files mid-write or in the old profile-only format
            zs.append(z)
        except Exception:
            pass
    return zs


def pooled(zs, key):
    tau = zs[0]['tau']
    return tau, [np.concatenate([z[key][i] for z in zs]) for i in range(len(tau))]


def ref_series(key):
    """Pooled ReaDDy reference: early + production windows, log-thinned."""
    tau, pools, nseed = [], [], {}
    for pat in ('rf_early_s*.npz', 'rf_s*.npz'):
        zs = load_all(sorted(glob.glob(os.path.join(RDY, pat))))
        if not zs:
            continue
        t, P = pooled(zs, key)
        tau.extend(t); pools.extend(P); nseed[pat] = len(zs)
    o = np.argsort(tau)
    tau = np.array(tau)[o]; pools = [pools[i] for i in o]
    pos = tau > 0
    targets = np.geomspace(tau[pos].min(), tau.max(), NLOG)
    keep = sorted({int(np.argmin(np.abs(tau - tg))) for tg in targets})
    return tau[keep], [pools[i] for i in keep], nseed


def lat_series(tag, key):
    tau, P = [], []
    for win in ('veryearly', 'early', 'late'):
        zs = load_all(sorted(glob.glob(os.path.join(LAT, '%s_%s_s*.npz' % (win, tag)))))
        if not zs:
            continue
        t, Pw = pooled(zs, key)
        tau.extend(t); P.extend([hist(p) for p in Pw])
    o = np.argsort(tau)
    return np.array(tau)[o], [P[i] for i in o]


def bd_series(key):
    zs = load_all(sorted(glob.glob(os.path.join(HERE, 'rxl80_out', 'bd_s*.npz')),
                         key=lambda f: int(re.search(r's(\d+)', f).group(1))))
    t, P = pooled(zs, key)
    return t, [hist(p) for p in P], len(zs)


def match(tref, Pref, tx, Px):
    """One model frame per reference frame, relative tolerance (see bd_sin_kl.py)."""
    x, y = [], []
    for i, t in enumerate(tref):
        j = int(np.argmin(np.abs(tx - t)))
        if t <= 0 or abs(tx[j] - t) > 0.15 * t:
            continue
        d = kl(Pref[i], Px[j])
        if d <= 0:
            continue
        x.append(t); y.append(d)
    return np.array(x), np.array(y)


def floor_series(tref, pools, nsub, ndraw=60, seed=0):
    """Same-distribution KL floor at sample size nsub, by resampling the reference pool."""
    rng = np.random.default_rng(seed)
    return np.array([np.mean([kl(hist(pool), hist(rng.choice(pool, size=nsub, replace=True)))
                              for _ in range(ndraw)]) for pool in pools])


def main():
    cache = {}
    for key, name in (('occA', 'A'), ('occC', 'C')):
        tr, pools, nseed = ref_series(key)
        Pref = [hist(p) for p in pools]
        arms = {'excl': lat_series('excl', key), 'ideal': lat_series('ideal', key)}
        for arm, (tx, Px) in arms.items():
            if len(tx) == 0:
                print('%s/%s: no data yet' % (name, arm)); continue
            x, y = match(tr, Pref, tx, Px)
            cache['%s_%s_x' % (name, arm)], cache['%s_%s_y' % (name, arm)] = x, y
            late = y[x >= 1.0]
            print('%s/%-5s %2d frames  KL range %.2e..%.2e  late-time mean %s'
                  % (name, arm, len(x), y.min() if len(y) else np.nan,
                     y.max() if len(y) else np.nan,
                     '%.2e' % late.mean() if len(late) else 'n/a'))
            for ts in SNAP_TAUS:
                j = int(np.argmin(np.abs(tx - ts)))
                if abs(tx[j] - ts) <= 0.3 * ts:
                    cache['snap_%s_%s_%g' % (name, arm, ts)] = Px[j]
        cache['floor_%s' % name] = floor_series(tr, pools, nsub=32768)
        cache['floor_tau'] = tr
        for ts in SNAP_TAUS:
            i = int(np.argmin(np.abs(tr - ts)))
            if abs(tr[i] - ts) <= 0.3 * ts:
                cache['snap_%s_ref_%g' % (name, ts)] = Pref[i]
        print('%s: reference %s seeds, %d frames kept' % (name, nseed, len(tr)))
    np.savez_compressed(os.path.join(HERE, 'kl3way_cache.npz'), **cache)
    print('cached ->', os.path.join(HERE, 'kl3way_cache.npz'))


if __name__ == '__main__':
    main()
