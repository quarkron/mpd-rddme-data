"""Recover the 10-seed replicate results from the driver logs and write one npz per (h, sweep).

The runs completed but np.savez could not write: figures_wm6/ is root-owned from an earlier
container, the host-side mkdir of rep/ was denied, and savez does not create directories. Every
per-run result is in the printed log line, so nothing was lost -- this parses those.
"""
import re, glob
import numpy as np

PAT = re.compile(
    r'^\s+(freq|tstep)\s+s=(\d+)\s+omega/Dk=\s*([\d.]+)\s+tau=\s*([\d.]+)us\s+'
    r'eps_samp=([\d.]+)\s+eps_tau=([\d.]+)\s+\|\s+A=\s*([-\d.]+)\s+\(an\s+([-\d.]+)\)\s+'
    r'lag=\s*([-\d.]+)\s+\(an\s+([-\d.]+)\)\s+w=\s*([-\d.]+)\s+\|\s+steps=(\d+)\s+'
    r'fitpts=(\d+)\s+frames=(\d+)\s+\|\s+WALL\s+([\d.]+)s')

rows = []
for f in sorted(glob.glob('/raid/racda/tmp/wm6rep/g?.log')):
    # each leg runs h=2 (freq+tstep) then h=4 (tstep only); eps_samp distinguishes them
    for line in open(f, errors='ignore'):
        m = PAT.match(line.rstrip('\n'))
        if not m:
            continue
        kind, seed, r, tau_us, es, et, A, Aan, lg, lgan, w, steps, nfit, nfr, wall = m.groups()
        es = float(es)
        # h = sqrt(D tau / eps_samp)/UNIT in box units; D=1e-12, UNIT=1e-8
        h = np.sqrt(1e-12 * float(tau_us) * 1e-6 / es) / 1e-8
        rows.append(dict(kind=kind, seed=int(seed), r=float(r), tau=float(tau_us) * 1e-6,
                         eps_samp=es, eps_tau=float(et), A=float(A), Aan=float(Aan),
                         lag=float(lg), lagan=float(lgan), w=float(w), h=round(h, 3),
                         steps=int(steps), wall=float(wall)))
print('parsed %d run records' % len(rows))
gpu_h = sum(x['wall'] for x in rows) / 3600.0
print('total compute: %.2f GPU-hours' % gpu_h)

def agg(sel, key_of):
    """Group by configuration, return mean and standard error of the mean over seeds."""
    out = {}
    for x in rows:
        if not sel(x):
            continue
        out.setdefault(key_of(x), []).append(x)
    res = []
    for k in sorted(out):
        g = out[k]
        A = np.array([x['A'] for x in g]); Lg = np.array([x['lag'] for x in g])
        n = len(g)
        res.append(dict(key=k, n=n, h=g[0]['h'], eps_tau=g[0]['eps_tau'],
                        eps_samp=g[0]['eps_samp'], tau=g[0]['tau'], r=g[0]['r'],
                        A=A.mean(), A_sem=A.std(ddof=1) / np.sqrt(n), Aan=g[0]['Aan'],
                        lag=Lg.mean(), lag_sem=Lg.std(ddof=1) / np.sqrt(n), lagan=g[0]['lagan']))
    return res

def dump(title, res):
    print('\n=== %s ===' % title)
    print('  key      n   eps_tau eps_samp |   A_mean +- SEM      A_an    dev%%   sig |'
          '   lag_mean +- SEM     lag_an   dev%%   sig')
    for d in res:
        dA = abs(d['A'] / d['Aan'] - 1) * 100; bA = d['A_sem'] / d['Aan'] * 100
        dL = abs(d['lag'] / d['lagan'] - 1) * 100; bL = d['lag_sem'] / d['lagan'] * 100
        print('  %-8s %2d  %.3f   %.3f  | %8.4f+-%.4f %8.3f %6.3f %5.1f | %7.3f+-%.4f %7.3f %6.3f %5.1f'
              % (d['key'], d['n'], d['eps_tau'], d['eps_samp'], d['A'], d['A_sem'], d['Aan'], dA,
                 dA / bA if bA else 0, d['lag'], d['lag_sem'], d['lagan'], dL, dL / bL if bL else 0))
    return res

FR = dump('frequency sweep, h=2', agg(lambda x: x['kind'] == 'freq', lambda x: 'om%.2f' % x['r']))
T2 = dump('timestep sweep, h=2',
          agg(lambda x: x['kind'] == 'tstep' and x['h'] == 2.0, lambda x: 'tau%.0f' % (x['tau'] * 1e6)))
T4 = dump('timestep sweep, h=4',
          agg(lambda x: x['kind'] == 'tstep' and x['h'] == 4.0, lambda x: 'tau%.0f' % (x['tau'] * 1e6)))

def tonp(res):
    return np.array([[d['r'], d['tau'], d['A'], d['Aan'], d['lag'], d['lagan'], 0.0,
                      d['eps_tau'], d['A_sem'], d['lag_sem'], d['n']] for d in res])

COLS = np.array(['omega_over_Dk', 'tau_s', 'amp', 'amp_analytic', 'lag_deg', 'lag_deg_analytic',
                 'width', 'eps_tau1_2sigma', 'amp_sem', 'lag_sem', 'n_seeds'])
BASE = '/home/racda/ftsz_tests/Lattice_Microbes_2.6/testing/mpd-crddme/wm6_replicates/'
np.savez(BASE + 'rep_h2.npz', freq=tonp(FR), tstep=tonp(T2), cols=COLS,
         meta=np.array([128, 2.0, 40e-6, 100.0, 10.0, 40.0, 0.10, 3.0]))
np.savez(BASE + 'rep_h4.npz', freq=np.zeros((0, 11)), tstep=tonp(T4), cols=COLS,
         meta=np.array([64, 4.0, 160e-6, 100.0, 10.0, 40.0, 0.10, 3.0]))
print('\n-> rep_h2.npz, rep_h4.npz')
