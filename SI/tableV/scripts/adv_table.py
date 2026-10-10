"""Main-text table: cost ratio, ceiling ratio and advantage for the ternary mixture and the
sinusoidal reacting configuration, at every hardware pairing that was measured.

Emits LaTeX to out/adv_table.tex and a readable version to stdout. Both carry the same numbers;
neither is hand-edited into the manuscript.

The ternary has no single-core lattice row on purpose. The single-file CPU reference implements
the byte-lattice species and the White Bear hop work, not the oversize-body machinery that carries
a sphere larger than a subvolume, so a core-to-core ternary comparison would be between two
different models. That is stated in the caption rather than papered over with a number.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adv_schemes as S
import adv_analyze as AN

RUNS = os.path.join(HERE, '..', 'data', 'runs')
OUT = os.path.join(HERE, '..', 'plots')
TAU_R = 2.0 / (S.D_NM2S * S.K_SOFT)

PAIRS = [('core_matched', 'one core, matched physics'),
         ('core_best', 'one core, best kernel'),
         ('socket', 'one socket each'),
         ('gpu_vs_core', 'GPU against one core'),
         ('gpu_vs_socket', 'GPU against one socket')]
LAT_OF = {'core_matched': 'lat_cpu1', 'core_best': 'lat_cpu1', 'socket': 'lat_cpu56',
          'gpu_vs_core': 'lat_gpu', 'gpu_vs_socket': 'lat_gpu'}
RD_OF = {'core_matched': 'rd_sc', 'core_best': 'rd_cpu1', 'socket': 'rd_cpu56',
         'gpu_vs_core': 'rd_sc', 'gpu_vs_socket': 'rd_cpu56'}
# The ternary files are named by the ReaDDy driver's own arm strings; map them onto the canonical
# keys the pairings use, or the ternary block silently produces no rows at all.
TERNARY_ARM = {'sc_db': 'rd_sc', 'cpu_g': 'rd_cpu1', 'cpu_g56': 'rd_cpu56'}


def ternary_rows():
    """Collect the ternary arms, which live in their own files rather than in the eta sweep."""
    import glob
    lat, rd = {}, {}
    for f in glob.glob(os.path.join(RUNS, 'ternary_lat_model_r*.json')):
        lat.setdefault('lat_gpu', []).append(json.load(open(f)))
    for f in glob.glob(os.path.join(RUNS, 'ternary_rd_*.json')):
        arm = os.path.basename(f).split('ternary_rd_')[1].rsplit('_r', 1)[0]
        key = TERNARY_ARM.get(arm)
        if key is None:
            continue
        rd.setdefault(key, []).append(json.load(open(f)))
    if not lat:
        return None
    g = lat['lat_gpu']
    tau_samp = AN.median([r['tau_samp_s'] for r in g])
    cost = {'lat_gpu': AN.median([r['per_step_s'] for r in g])}
    for k, v in rd.items():
        cost[k] = AN.median([r['per_step_s'] for r in v])
    return dict(system='ternary mixture', tau_samp_s=tau_samp, tau_rxn_s=float('inf'),
                tau_L_s=tau_samp, cost=cost, n0=g[0]['n0'], note='oversize body, no reactions')


def sinusoidal_row():
    rows = json.load(open(os.path.join(RUNS, 'summary.json')))
    cand = [r for r in rows if r['scheme'] == 'rev']
    if not cand:
        return None
    # the reference crowding of the published reacting configuration
    r = min(cand, key=lambda r: abs(r['eta'] - 0.09))
    return dict(system='sinusoidal, reversible binding', tau_samp_s=r['tau_samp_s'],
                tau_rxn_s=r['tau_rxn_s'], tau_L_s=r['tau_L_s'], cost=r['cost'], n0=r['n0'],
                note='eta = %.2f' % r['eta'])


def fmt(x, n=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '--'
    if x >= 1000:
        e = int(math.floor(math.log10(x)))
        return r'$%.1f\times10^{%d}$' % (x / 10 ** e, e)
    return ('%.*f' % (n, x)).rstrip('0').rstrip('.') if x < 100 else '%.0f' % x


def main():
    systems = [s for s in (ternary_rows(), sinusoidal_row()) if s]
    if not systems:
        print('no data yet'); return
    os.makedirs(OUT, exist_ok=True)
    lines = []
    plain = []
    for sysrow in systems:
        cost = sysrow['cost']
        ratio_ceiling = sysrow['tau_L_s'] / TAU_R
        first = True
        for key, lab in PAIRS:
            lat, rd = LAT_OF[key], RD_OF[key]
            if lat not in cost or rd not in cost or cost[lat] <= 0:
                continue
            cr = cost[rd] / cost[lat]
            A = cr * ratio_ceiling
            name = sysrow['system'] if first else ''
            tl = fmt(sysrow['tau_L_s'] * 1e6, 1) if first else ''
            rc = fmt(ratio_ceiling, 0) if first else ''
            lines.append('%s & %s & %s & %s & %s & %s \\\\' %
                         (name, lab, fmt(cr), tl, rc, fmt(A, 0)))
            plain.append('%-30s %-26s cost %8.3g  tau_L %6.1f us  ceiling %7.0f  A %9.3g'
                         % (sysrow['system'] if first else '', lab, cr,
                            sysrow['tau_L_s'] * 1e6, ratio_ceiling, A))
            first = False
        lines.append(r'\addlinespace')
    if lines and lines[-1] == r'\addlinespace':
        lines.pop()

    tex = [r'\begin{table}[htbp]', r'  \centering',
           r'  \caption{Cost ratio, timestep-ceiling ratio and the resulting advantage for the two',
           r'  main-text configurations. The cost ratio is the measured wall time per step of the',
           r'  particle engine over that of the lattice at the same box and the same particle',
           r'  content. The ceiling ratio is the largest step each method may take at a declared',
           r'  accuracy budget, with the particle engine quoted at its Euler stability limit',
           r'  rather than at the smaller step it is configured with. Matched physics means the',
           r'  detailed-balance reaction handler, which the particle engine implements for its',
           r'  single-core kernel only; best kernel means its multithreaded kernel with the',
           r'  standard handler, which is faster and is a different reaction model. The ternary',
           r'  mixture carries a sphere larger than a subvolume, which the single-core lattice',
           r'  reference does not implement, so that system is reported on the device arm only.}',
           r'  \label{tab:advantage}',
           r'  \begin{tabular}{llrrrr}', r'  \toprule',
           r'  system & pairing & $c_{\mathrm{R}}/c_{\mathrm{L}}$ & $\tau_{\mathrm{L}}$ ($\mu$s)'
           r' & $\tau_{\mathrm{L}}/\tau_{\mathrm{R}}$ & $A$ \\', r'  \midrule']
    tex += ['  ' + l for l in lines]
    tex += [r'  \bottomrule', r'  \end{tabular}', r'\end{table}']
    open(os.path.join(OUT, 'adv_table.tex'), 'w').write('\n'.join(tex) + '\n')
    print('\n'.join(plain))
    print('\nwrote %s/adv_table.tex' % OUT)


if __name__ == '__main__':
    main()
