"""EXPERIMENT (2026-10-08): which demixed start gives block d of fig:polymer the most striking contrast?

    python fig4d_noneq_compare.py [--root /raid/racda/lmrddme_polymer/demix_noneq] [--times 0,1,10]

For each start in gradient_slab.KINDS (runs made by demix_slab.sh, one folder per start) it draws the crowder density
c / mean(c) in the middle slab at three times, finite-size over point particles (one tone, white -> crowder blue,
one colour axis for every map), and scores the mixing: the fraction of crowders, and of chain subunits, still inside the
region they started in, at every snapshot (1 at the start; 0.5 = fully mixed, since each region is half the box).
Writes ../plots/experiments/fig4d_noneq_maps.{pdf,png} and fig4d_noneq_retention.{pdf,png}.
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, '/home/racda/ftsz_tests/studies/polymer')
import fig_polymer_r2 as F                      # noqa: E402
import gradient_slab as GS                      # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/raid/racda/lmrddme_polymer/demix_noneq')
ap.add_argument('--times', default='0,1,10')
A = ap.parse_args()
want = [float(t) for t in A.times.split(',')]
OUT = os.path.join(HERE, '..', 'plots', 'experiments')


def load(kind, run):
    return np.load(os.path.join(A.root, kind, run + '.npz'), allow_pickle=True)


def crowd_field(d, k):
    nx = int(d['arg_nx']); bath = 'snap_bath_%d' % k
    C, nc = (F._field(d[bath], nx) if bath in d.files else F._blocks(d['snap_body_%d' % k], nx))
    return C, nc


def crowd_slice(d, k, cut_axis=1):
    C, nc = crowd_field(d, k)
    k0 = nc // 2 - F.SLAB // 2
    sl = [slice(None)] * 3; sl[cut_axis] = slice(k0, k0 + F.SLAB)
    c = C[tuple(sl)].mean(axis=cut_axis) / max(C.mean(), 1e-12)
    return c   # no rim crop: these starts have real structure at the faces


def retention(d, kind):
    """Per snapshot: (t, fraction of crowders in the crowder region, fraction of chain subunits in the monomer region)."""
    nx = int(d['arg_nx']); m = GS.crowder_mask(nx, kind)
    ks = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_t_'))
    out = []
    for k in ks:
        bath = 'snap_bath_%d' % k
        if bath in d.files:
            c = np.asarray(d[bath], float).reshape(-1); fc = c[m].sum() / max(c.sum(), 1)
        else:
            b = np.asarray(d['snap_body_%d' % k], np.int64); fc = m[b].mean() if len(b) else np.nan
        ch = np.asarray(d['snap_chain_%d' % k], np.int64); fp = (~m[ch]).mean() if len(ch) else np.nan
        out.append((float(d['snap_t_%d' % k]), fc, fp))
    return np.array(out)


def snap_index(d, t):
    ks = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_t_'))
    return ks[int(np.argmin([abs(float(d['snap_t_%d' % k]) - t) for k in ks]))]


rc, _ = F.style()
kinds = [k for k in GS.KINDS if os.path.exists(os.path.join(A.root, k, F.ARMS_D[0][0] + '.npz'))]
with plt.rc_context(rc):
    maps = {(kd, run): [crowd_slice(load(kd, run), snap_index(load(kd, run), t)) for t in want] for kd in kinds for run, _ in F.ARMS_D}
    vmax = float(np.ceil(max(np.percentile(f, 99) for v in maps.values() for f in v) * 4) / 4)
    cmap = LinearSegmentedColormap.from_list('crowd', ['#ffffff', F.C_CROWD])
    nt = len(want); W = 0.9 + len(kinds) * (nt * 0.95 + 0.35); H = 2.6
    fig = plt.figure(figsize=(W, H))
    for g, kd in enumerate(kinds):
        x0 = 0.55 + g * (nt * 0.95 + 0.35)
        fig.text((x0 + nt * 0.95 / 2) / W, 0.965, kd, ha='center', va='top', fontsize=8.5, fontweight='bold')
        for i, (run, name) in enumerate(F.ARMS_D):
            for j, t in enumerate(want):
                ax = fig.add_axes([(x0 + j * 0.95) / W, (1.30 - 1.05 * i) / H, 0.88 / W, 0.88 / H])
                im = ax.imshow(maps[(kd, run)][j].T, origin='lower', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_linewidth(0.5); sp.set_color(F.KP.CAGE)
                if i == 0: ax.set_title(r'$t=%g$ s' % t, fontsize=7, pad=2)
                if g == 0 and j == 0: ax.text(-0.1, 0.5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7)
    cax = fig.add_axes([(W - 0.3) / W, 0.25 / H, 0.07 / W, 1.95 / H])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r'crowder density $c/\bar c$', fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2)
    for e in ('pdf', 'png'): fig.savefig(os.path.join(OUT, 'fig4d_noneq_maps.' + e), dpi=220)

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.3))
    cols = dict(zip(GS.KINDS, ['#123f7a', '#2bb3a3', '#d97321', '#9cbf3c']))
    print('%-8s %-16s %s' % ('start', 'arm', 'crowders in region at t = 0,1,2,5,10 | chain subunits in region at t = 1,2,5,10'))
    for kd in kinds:
        for (run, name), ls in zip(F.ARMS_D, ('-', '--')):
            r = retention(load(kd, run), kd)
            axs[0].plot(r[:, 0], r[:, 1], ls, color=cols[kd], marker='o', ms=2.5, lw=1.1, label='%s, %s' % (kd, name))
            axs[1].plot(r[:, 0], r[:, 2], ls, color=cols[kd], marker='o', ms=2.5, lw=1.1)
            pick = lambda col, ts: ' '.join('%.3f' % r[int(np.argmin(abs(r[:, 0] - t))), col] for t in ts)
            print('%-8s %-16s %s | %s' % (kd, name, pick(1, (0, 1, 2, 5, 10)), pick(2, (1, 2, 5, 10))))
    for a, lab in zip(axs, ('crowders in their starting region', 'chain subunits in the monomer region')):
        a.axhline(0.5, color='0.6', lw=0.7, ls=':'); a.set_ylim(0.4, 1.02); a.set_xlabel('time (s)'); a.set_ylabel(lab, fontsize=7.5)
        a.grid(alpha=0.25, lw=0.4)
    axs[0].legend(fontsize=5.8, frameon=False, ncol=2, loc='lower left')
    fig.tight_layout()
    for e in ('pdf', 'png'): fig.savefig(os.path.join(OUT, 'fig4d_noneq_retention.' + e), dpi=220)
print('wrote', os.path.normpath(OUT) + '/fig4d_noneq_{maps,retention}.{pdf,png}')
