"""EXPERIMENT (2026-10-09): block d as one-tone POLYMER maps from demixed starts (see fig4d_noneq_compare.py).

Polymer material per cell relative to its box mean: free monomers at t = 0 (the .lm initial lattice; no chain exists
yet), chain subunits afterwards (snap_chain; free monomers are not stored in the snapshots). One ramp, white -> polymer
green, one colour axis for every map. Writes ../plots/experiments/fig4d_noneq_polymer.{pdf,png}.
    python fig4d_noneq_polymer.py [--kinds slab,droplet] [--times 0,1,10]
"""
import argparse, os, sys
import numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_polymer_r2 as F                      # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/raid/racda/lmrddme_polymer/demix_noneq')
ap.add_argument('--kinds', default='slab,droplet'); ap.add_argument('--times', default='0,1,10')
A = ap.parse_args(); kinds = A.kinds.split(','); want = [float(t) for t in A.times.split(',')]


def poly_slice(kind, run, t, cut_axis=1):
    d = np.load(os.path.join(A.root, kind, run + '.npz'), allow_pickle=True); nx = int(d['arg_nx'])
    if t == 0:
        L = h5py.File(os.path.join(A.root, kind, run + '.lm'), 'r')['Simulations/0000001/Lattice/0000000000'][:]
        m = (L == 1).sum(axis=3).astype(float)                              # [x, y, z] count of species M (id 1)
        flat = np.transpose(m, (2, 1, 0)).reshape(-1)                       # -> v = z*nx*nx + y*nx + x, like the snapshots
        P, nc = F._field(flat, nx); tt = 0.0
    else:
        ks = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_t_'))
        k = ks[int(np.argmin([abs(float(d['snap_t_%d' % q]) - t) for q in ks]))]
        P, nc = F._blocks(d['snap_chain_%d' % k], nx); tt = float(d['snap_t_%d' % k])
    k0 = nc // 2 - F.SLAB // 2
    sl = [slice(None)] * 3; sl[cut_axis] = slice(k0, k0 + F.SLAB)
    return P[tuple(sl)].mean(axis=cut_axis) / max(P.mean(), 1e-12), tt


rc, _ = F.style()
with plt.rc_context(rc):
    maps = {(kd, run): [poly_slice(kd, run, t) for t in want] for kd in kinds for run, _ in F.ARMS_D}
    vmax = 3.0                                    # saturate the densest clumps so the domains read against white
    cmap = LinearSegmentedColormap.from_list('poly', ['#ffffff', F.C_POLY])
    nt = len(want); W = 0.9 + len(kinds) * (nt * 1.05 + 0.35); H = 2.75
    fig = plt.figure(figsize=(W, H))
    for g, kd in enumerate(kinds):
        x0 = 0.55 + g * (nt * 1.05 + 0.35)
        fig.text((x0 + nt * 1.05 / 2) / W, 0.97, '%s start' % kd, ha='center', va='top', fontsize=8.5, fontweight='bold')
        for i, (run, name) in enumerate(F.ARMS_D):
            for j, _ in enumerate(want):
                f, tt = maps[(kd, run)][j]
                ax = fig.add_axes([(x0 + j * 1.05) / W, (1.36 - 1.10 * i) / H, 0.97 / W, 0.97 / H])
                im = ax.imshow(f.T, origin='lower', cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_linewidth(0.5); sp.set_color(F.KP.CAGE)
                if i == 0: ax.set_title(r'$t=%g$ s' % round(tt), fontsize=7.5, pad=2)
                if g == 0 and j == 0: ax.text(-0.1, 0.5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.5)
                print('  %-8s %-16s t=%4.1f  polymer map sd %.3f' % (kd, name, tt, f.std()))
    cax = fig.add_axes([(W - 0.3) / W, 0.25 / H, 0.07 / W, 2.1 / H])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r'polymer density $p/\bar p$', fontsize=7.5); cb.ax.tick_params(labelsize=6.5, length=2)
    out = os.path.join(HERE, '..', 'plots', 'experiments', 'fig4d_noneq_polymer')
    for e in ('pdf', 'png'): fig.savefig(out + '.' + e, dpi=220)
print('wrote', os.path.normpath(out) + '.{pdf,png}')
