"""Turn the campaign's raw timings into ceilings, advantage factors and the normalised metric.

The metric, as derived in the manuscript note notes/perf_01_timestep_ceilings.pdf:

    A(eta) = [c_R/c_L] x [tau_L/tau_R],      tau_L = min(tau_samp, tau_rxn)

Every factor is measured rather than assumed:
  c        median per-step wall time over repetitions, from two-point differencing.
  tau_samp lattice transport ceiling, dt_ceiling * tau_scale from the adaptive-tau controller.
  tau_rxn  lattice reaction ceiling, eps_r / max_v a_v, with a_v measured on the occupancy field.
  tau_R    ReaDDy transport ceiling, the Euler-Maruyama stability limit 2/(beta D k). Quoted
           against the LIMIT, not against the step the engine is configured at, which is a factor
           of 2.5 inside it; both are recorded.

The normalised metric divides out the PREDICTED ceiling ratio, using the extreme-value estimate of
the extremal Bernoulli factor built from the measured occupancy histogram alone. If the ceiling
analysis is right, A_hat must collapse onto the independently measured cost ratio at every
crowding level. Drift of one against the other refutes it, and that is the test.
"""
import json, glob, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adv_schemes as S

RUNS = os.path.join(HERE, '..', 'data', 'runs')
EPS_R = 0.1          # declared tolerance on the single-firing truncation of the reaction kernel


def B(u):
    return 1.0 if abs(u) < 1e-12 else u / math.expm1(u)


def ndtri(p):
    """Inverse standard normal CDF (Acklam), for the occupancy quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def cs_fex(n, eta1):
    """Carnahan-Starling excess free energy of n spheres each contributing packing eta1."""
    e = n * eta1
    if e >= 1.0:
        return float('inf')
    return n * e * (4.0 - 3.0 * e) / (1.0 - e) ** 2


def predicted_maxB(occ_mean, occ_var, eta, V):
    """A priori extreme-value estimate of the per-direction extremal Bernoulli factor.

    Inputs are the MEASURED occupancy mean and variance and the packing fraction; nothing is
    fitted. The crowded end of a hop sits near the (1 - 1/3V) quantile of the occupancy
    distribution, since a lattice of V subvolumes carries 3V nearest-neighbour bonds.
    """
    if occ_mean <= 0:
        return 1.0, occ_mean
    eta1 = eta / occ_mean                      # packing contributed per particle
    z = ndtri(1.0 - 1.0 / (3.0 * V))
    n_hi = occ_mean + z * math.sqrt(max(occ_var, 1e-9))
    dphi = 2.0 * S.GAMMA * math.sin(math.pi * S.H_NM / S.WAVELENGTH)
    dU = ((cs_fex(occ_mean + 1, eta1) - cs_fex(occ_mean, eta1))
          - (cs_fex(n_hi, eta1) - cs_fex(n_hi - 1, eta1)) - dphi)
    return B(dU), n_hi


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float('nan')


def load():
    pts = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(RUNS, '*.json'))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        base = os.path.basename(f)
        if '_eta' not in base:
            continue
        sc = base.split('_eta')[0]
        eta = float(base.split('_eta')[1].split('_')[0])
        arm = base.split('_eta%.3f_' % eta)[1].rsplit('_r', 1)[0]
        pts[(sc, eta)][arm].append(d)
    return pts


def summarize():
    pts = load()
    tau_R_stab = 2.0 / (S.D_NM2S * S.K_SOFT)
    rows = []
    for (sc, eta) in sorted(pts):
        arms = pts[(sc, eta)]
        cost = {a: median([r['per_step_s'] for r in rs]) for a, rs in arms.items()}
        # Relative spread of the repetitions, max-min over median. A timing ratio quoted without
        # it is not reportable, and it is the only signal that a run met a contended machine.
        spread = {}
        for a, rs in arms.items():
            v = sorted(r['per_step_s'] for r in rs)
            spread[a] = (v[-1] - v[0]) / v[len(v) // 2] if v and v[len(v) // 2] > 0 else float('nan')
        nrep = {a: len(rs) for a, rs in arms.items()}
        gpu = arms.get('lat_gpu', [])
        if not gpu:
            continue
        tau_samp = median([r['tau_samp_s'] for r in gpu])
        a_max = median([r['a_max_per_s'] for r in gpu])
        occ_mean = median([r['occ_mean'] for r in gpu])
        occ_var = median([r['occ_var'] for r in gpu])
        occ_max = median([r['occ_max'] for r in gpu])
        V = gpu[0]['M']
        tau_rxn = (EPS_R / a_max) if a_max > 0 else float('inf')
        tau_L = min(tau_samp, tau_rxn)
        Bpred, n_hi = predicted_maxB(occ_mean, occ_var, eta, V)
        A_eps = gpu[0]['eps']
        # measured per-direction factor, inverted from the controller: sum' = q*ts*(B- + B+) = eps
        q = S.D_NM2S * gpu[0]['dt_ceiling'] / S.H_NM ** 2
        ts = median([r['tau_scale'] for r in gpu])
        Bmeas = gpu[0]['eps'] / (q * ts) / 2.0
        # Predicted ceiling ratio, at THE SAME CRITERION the engine measures at, so that A_hat is
        # a test and not a unit conversion. ★The engine's budget eps bounds the SUM of the two
        # directions out of a subvolume, B- + B+ ~ 2B, whereas the closed form of the manuscript
        # note is written for the one-directional maximum at eps = 1/2. Mixing the two costs a
        # factor of (1/2)/(eps/2) = 5 at eps = 0.2, which is exactly the offset that showed up
        # between A_hat and the measured cost ratio the first time this was computed.
        #     tau_samp(eps) = eps h^2 / (D (B- + B+)) = (eps/2) h^2 / (D B)
        #     tau_R         = 2 / (beta D k)
        #     ratio         = eps beta k h^2 / (4 B)          <- D cancels
        ratio_pred = A_eps * S.K_SOFT * S.H_NM ** 2 / (4.0 * Bpred)
        row = dict(scheme=sc, eta=eta, V=V, n0=gpu[0]['n0'],
                   cost=cost, tau_samp_s=tau_samp, tau_rxn_s=tau_rxn, tau_L_s=tau_L,
                   tau_R_stab_s=tau_R_stab, binding=('rxn' if tau_rxn < tau_samp else 'samp'),
                   a_max_per_s=a_max, occ_mean=occ_mean, occ_var=occ_var, occ_max=occ_max,
                   B_meas=Bmeas, B_pred=Bpred, n_hi=n_hi, ratio_pred=ratio_pred,
                   ratio_meas=tau_samp / tau_R_stab, eps=A_eps,
                   cost_spread=spread, nrep=nrep,
                   tau_samp_spread=((max(r['tau_samp_s'] for r in gpu)
                                     - min(r['tau_samp_s'] for r in gpu)) / tau_samp
                                    if tau_samp > 0 else float('nan')))
        for rd, lat, name in (('rd_sc', 'lat_cpu1', 'core_matched'),
                              ('rd_cpu1', 'lat_cpu1', 'core_best'),
                              ('rd_cpu56', 'lat_cpu56', 'socket'),
                              ('rd_cpu56', 'lat_gpu', 'gpu_vs_socket'),
                              ('rd_sc', 'lat_gpu', 'gpu_vs_core'),
                              # Best single-core particle kernel against each lattice arm in turn:
                              # the pair the supplement figure plots for all three schemes.
                              ('rd_cpu1', 'lat_gpu', 'bestcore_vs_gpu')):
            if rd in cost and lat in cost and cost[lat] > 0:
                cr = cost[rd] / cost[lat]
                A = cr * tau_L / tau_R_stab
                row['cost_ratio_' + name] = cr
                row['A_' + name] = A
                row['Ahat_' + name] = A / ratio_pred
        rows.append(row)
    return rows


def main():
    rows = summarize()
    out = os.path.join(RUNS, 'summary.json')
    json.dump(rows, open(out, 'w'), indent=1)
    if not rows:
        print('no complete points yet'); return
    print('%-5s %-5s %7s %9s %9s %6s | %7s %7s %6s | %9s %9s %9s'
          % ('sch', 'eta', 'n0', 'tsamp/us', 'trxn/us', 'bind', 'B_meas', 'B_pred', 'ratio',
             'c_core', 'A_core', 'Ahat_core'))
    for r in rows:
        print('%-5s %-5.3f %7d %9.1f %9.3g %6s | %7.2f %7.2f %6.2f | %9.3g %9.3g %9.3g'
              % (r['scheme'], r['eta'], r['n0'], r['tau_samp_s'] * 1e6,
                 r['tau_rxn_s'] * 1e6 if r['tau_rxn_s'] != float('inf') else float('inf'),
                 r['binding'], r['B_meas'], r['B_pred'],
                 r['B_pred'] / r['B_meas'] if r['B_meas'] else float('nan'),
                 r.get('cost_ratio_core_matched', float('nan')),
                 r.get('A_core_matched', float('nan')),
                 r.get('Ahat_core_matched', float('nan'))))
    print('\nwrote %s (%d points)' % (out, len(rows)))


if __name__ == '__main__':
    main()
