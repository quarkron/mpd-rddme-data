"""EXPERIMENT (2026-10-08): block d of fig:polymer as one-tone maps at three times, to sharpen the contrast between the
finite-size and the point-particle arm. Not the figure of record (fig_polymer_r2.py is).

    python fig4d_onetone.py [--runs DIR] [--times 0,1,10] [--quantity share|crowd]

Same slices as fig_polymer_r2.snap_slice (middle 40 nm slab, 20 nm cells smoothed over 30 nm, rim cropped), one
colour ramp from white:
  share  polymer share p/(p + c), p and c each normalised by its own box mean: 0.5 = the box-average mixture,
         1 = all polymer. White at 0 (t = 0: no chain exists yet). Ramp white -> polymer green.
  crowd  crowder density c / mean(c): 1 = uniform. Ramp white -> crowder blue, from 0 to the 99th percentile.
Writes ../plots/experiments/fig4d_onetone_<quantity>.{pdf,png}.
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_polymer_r2 as F                      # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default=F.DEMIX)
ap.add_argument('--times', default='0,2,10')
ap.add_argument('--quantity', choices=('share', 'crowd'), default='share')
A = ap.parse_args()
want = [float(t) for t in A.times.split(',')]


def fields(run, k, cut_axis=1):
    d = np.load(os.path.join(A.runs, run + '.npz'), allow_pickle=True)
    nx = int(d['arg_nx'])
    P, nc = F._blocks(d['snap_chain_%d' % k], nx)
    bath = 'snap_bath_%d' % k
    C, _ = (F._field(d[bath], nx) if bath in d.files else F._blocks(d['snap_body_%d' % k], nx))
    k0 = nc // 2 - F.SLAB // 2
    sl = [slice(None)] * 3; sl[cut_axis] = slice(k0, k0 + F.SLAB)
    p = P[tuple(sl)].mean(axis=cut_axis); c = C[tuple(sl)].mean(axis=cut_axis)
    cr = slice(F.CROP, -F.CROP)
    p, c = p[cr, cr], c[cr, cr]
    if A.quantity == 'share':
        pn = p / max(P.mean(), 1e-12); cn = c / max(C.mean(), 1e-12)
        v = pn / (pn + cn + 1e-12)
    else:
        v = c / max(C.mean(), 1e-12)
    return v, float(np.asarray(d['snap_t_%d' % k]))


def snap_index(run, t):
    d = np.load(os.path.join(A.runs, run + '.npz'), allow_pickle=True)
    ks = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_t_'))
    ts = [float(d['snap_t_%d' % k]) for k in ks]
    return ks[int(np.argmin(np.abs(np.array(ts) - t)))]


rc, _tex = F.style()
with plt.rc_context(rc):
    rows = [[fields(run, snap_index(run, t)) for t in want] for run, _ in F.ARMS_D]
    if A.quantity == 'share':
        cmap = LinearSegmentedColormap.from_list('share', ['#ffffff', F.C_POLY]); vmin, vmax, lab = 0.0, 1.0, r'polymer share $p/(p+c)$'
    else:
        cmap = LinearSegmentedColormap.from_list('crowd', ['#ffffff', F.C_CROWD]); vmin = 0.0
        vmax = float(np.ceil(max(np.percentile(f, 99) for row in rows for f, _ in row) * 4) / 4); lab = r'crowder density $c/\bar c$'
    n = len(want)
    fig = plt.figure(figsize=(0.5 + 1.25 * n + 0.9, 2.95))
    axs = [[fig.add_axes([(0.55 + 1.25 * j) / fig.get_figwidth(), (1.55 - 1.30 * i) / 2.95, 1.15 / fig.get_figwidth(), 1.15 / 2.95])
            for j in range(n)] for i in range(2)]
    for i, (row, (run, name)) in enumerate(zip(rows, F.ARMS_D)):
        for j, (v, t) in enumerate(row):
            ax = axs[i][j]
            im = ax.imshow(v.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color(F.KP.CAGE)
            if i == 0:
                ax.set_title(r'$t = %g$ s' % round(t), pad=2.5, fontsize=8)
            if j == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.5)
            print('  %-16s t=%5.1f  mean %.3f  sd %.3f  p1-p99 %.2f-%.2f' % (name, t, v.mean(), v.std(), *np.percentile(v, [1, 99])))
    cax = fig.add_axes([(0.55 + 1.25 * n + 0.1) / fig.get_figwidth(), 0.25 / 2.95, 0.08 / fig.get_figwidth(), 2.45 / 2.95])
    cb = fig.colorbar(im, cax=cax); cb.set_label(lab, fontsize=7.5); cb.ax.tick_params(labelsize=7, length=2)
    out = os.path.join(HERE, '..', 'plots', 'experiments', 'fig4d_onetone_%s' % A.quantity)
    for e in ('pdf', 'png'):
        fig.savefig(out + '.' + e, dpi=250)
print('wrote', os.path.normpath(out) + '.{pdf,png}')
