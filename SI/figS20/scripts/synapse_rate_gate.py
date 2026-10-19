"""Gate the bimolecular rate against the well-mixed kinetics it is supposed to reproduce.

    python synapse_run.py --r-core 0 --f0 0 ... --out out/rate.npz
    python synapse_rate_gate.py out/rate.npz [...]

WHY THIS EXISTS
    A second-order rate handed to a lattice solver can come out wrong by a constant factor in
    several independent ways: the M^-1 s^-1 to per-subvolume conversion, the subvolume volume
    versus the box volume, and adaptive tau rescaling propensities while the clock advances by
    the bare tau. The last one has cost this codebase an entire campaign before. None of them
    change the SHAPE of N_C(t), so none of them are visible without an absolute reference.

    With the field off and the whole disk seeded uniformly, the lattice must reproduce

        dN_C/dt = (k_on / (V N_A)) N_TCR N_pMHC - k_off N_C,

    with V the volume of the seeded slab. That is the comparison made here.
"""
import json, math, sys
import numpy as np

NA = 6.02214076e23


def ode(n_tcr0, n_pmhc0, k_on, k_off, vol_L, times):
    """Integrate the well-mixed kinetics on a fine grid and sample at `times`."""
    kf = k_on / (vol_L * NA)                     # per molecule per second
    t_end = float(times[-1])
    n = 200000
    dt = t_end / n
    a, b, c = float(n_tcr0), float(n_pmhc0), 0.0
    out, j = [], 0
    for i in range(n + 1):
        t = i * dt
        while j < len(times) and times[j] <= t + 0.5 * dt:
            out.append(c); j += 1
        r = kf * a * b - k_off * c
        a -= r * dt; b -= r * dt; c += r * dt
    while len(out) < len(times):
        out.append(c)
    return np.array(out)


def gate(path, tol=0.05):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    names = [str(x) for x in d['names']]
    ic = names.index('TCR_pMHC')
    it, im = names.index('TCR'), names.index('pMHC')
    times = d['times']
    # Seeded slab volume: voxels that ever hold anything, times the voxel volume.
    h_um = meta['h_nm'] * 1e-3
    vol_um3 = float(d['vox_per_bin'].sum()) * h_um ** 3
    vol_L = vol_um3 * 1e-15
    n0_t = d['total'][0, it] + d['total'][0, ic]
    n0_m = d['total'][0, im] + d['total'][0, ic]
    import synapse_model as M
    pred = ode(n0_t, n0_m, M.K_ON_TCR, M.K_OFF_TCR, vol_L, times)
    meas = d['total'][:, ic]
    print('%s  tau_scale=%.4g  adaptive_tau=%s  V=%.4f um^3'
          % (path, meta.get('tau_scale', float('nan')), meta.get('adaptive_tau', '?'), vol_um3))
    print('    %8s %12s %12s %8s' % ('t (s)', 'measured', 'well-mixed', 'ratio'))
    ratios = []
    for i in range(len(times)):
        r = meas[i] / pred[i] if pred[i] > 0 else float('nan')
        if i:
            ratios.append(r)
        print('    %8.4f %12.0f %12.1f %8.3f' % (times[i], meas[i], pred[i], r))
    med = float(np.nanmedian(ratios)) if ratios else float('nan')
    ok = abs(med - 1.0) <= tol
    print('    median ratio %.4f  ->  %s' % (med, 'PASS' if ok else 'FAIL'))
    return ok, med


if __name__ == '__main__':
    results = [gate(p) for p in sys.argv[1:]]
    print()
    for p, (ok, med) in zip(sys.argv[1:], results):
        print('%-28s %s  (x%.3f)' % (p, 'PASS' if ok else 'FAIL', med))
    sys.exit(0 if all(ok for ok, _ in results) else 1)
