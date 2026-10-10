"""Block d of fig:polymer from a demixed start (2026-10-09): slices through the middle of the box at a few times,
finite-size over point particles, in one of two colourings.

  onetone   polymer density p / mean(p), white -> polymer green
  twotone   the original block-d contrast (p - c)/(p + c) of eq:localAmount, crowder blue -> white -> polymer green

p is polymer material: free monomers at t = 0 (the .lm initial lattice -- no chain exists yet), chain subunits after
(snap_chain; the snapshots do not store free monomers). c is the crowders (snap_body for finite-size bodies, snap_bath
for point crowders). Same slab, cells and smoothing as fig_polymer_r2.snap_slice; no rim crop, because the demixed
start has real structure at the faces. The runs come from demix_slab.sh (--start slab|droplet).
"""
import os

import h5py
import numpy as np

import fig_polymer_r2 as F


def _snap(d, t):
    ks = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_t_'))
    return ks[int(np.argmin([abs(float(d['snap_t_%d' % q]) - t) for q in ks]))]


def fields(root, run, t):
    """(P, C, nc, t): smoothed polymer and crowder cell fields at the snapshot nearest t."""
    d = np.load(os.path.join(root, run + '.npz'), allow_pickle=True)
    nx = int(d['arg_nx'])
    k = _snap(d, t)
    if float(d['snap_t_%d' % k]) == 0.0:
        L = h5py.File(os.path.join(root, run + '.lm'), 'r')['Simulations/0000001/Lattice/0000000000'][:]
        m = np.transpose((L == 1).sum(axis=3).astype(float), (2, 1, 0)).reshape(-1)   # species M, v = z*nx*nx + y*nx + x
        P, nc = F._field(m, nx)
    else:
        P, nc = F._blocks(d['snap_chain_%d' % k], nx)
    bath = 'snap_bath_%d' % k
    C, _ = (F._field(d[bath], nx) if bath in d.files else F._blocks(d['snap_body_%d' % k], nx))
    return P, C, nc, float(d['snap_t_%d' % k])


def slice_of(root, run, t, mode, cut_axis=1):
    P, C, nc, tt = fields(root, run, t)
    k0 = nc // 2 - F.SLAB // 2
    sl = [slice(None)] * 3; sl[cut_axis] = slice(k0, k0 + F.SLAB)
    p = P[tuple(sl)].mean(axis=cut_axis) / max(P.mean(), 1e-12)
    if mode == 'onetone':
        return p, tt
    c = C[tuple(sl)].mean(axis=cut_axis) / max(C.mean(), 1e-12)
    return (p - c) / (p + c + 1e-12), tt


def panel_maps(grid, cax, root, arms, times, mode='onetone', titles=True, labels=True, frame=None, frame_lw=0.8, orient='vertical'):
    """grid[row][col] axes: rows = arms ((run, label), ...), columns = times. titles/labels: the t = and arm text;
    frame: spine colour (default the cage grey)."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    maps = [[slice_of(root, run, t, mode) for t in times] for run, _ in arms]
    if mode == 'onetone':
        cmap = LinearSegmentedColormap.from_list('poly', ['#ffffff', F.C_POLY]); vmin, vmax = 0.0, 3.0
        lab = r'polymer density $p/\bar p$'
    else:
        cmap = F.DIVERGING; vmin, vmax = -1.0, 1.0
        lab = r'$(p-c)/(p+c)$'
    im = None
    for r, (row, (run, name)) in enumerate(zip(maps, arms)):
        for c, (ax, (f, tt)) in enumerate(zip(grid[r], row)):
            im = ax.imshow(f.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6 if frame is None else frame_lw); sp.set_color(frame or F.KP.CAGE)
            if titles and r == 0:
                ax.set_title(r'$t = %g$ s' % round(tt), pad=2.5, fontsize=8)
            if labels and c == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.5,
                        color=F.INK)
            print('  %-16s t=%4.1f  %s map: mean %.3f sd %.3f' % (name, tt, mode, f.mean(), f.std()))
    cb = cax.figure.colorbar(im, cax=cax, orientation=orient)
    cb.ax.tick_params(labelsize=7, length=2)
    if orient == 'horizontal':                    # label above the bar, the end words under the end ticks
        cb.ax.set_title(lab, fontsize=7.5, pad=3)
        cb.set_ticks([vmin, 0.5 * (vmin + vmax), vmax])
        if mode == 'twotone':
            for x, txt in ((0.0, 'crowder-rich'), (1.0, 'polymer-rich')):
                cb.ax.text(x, -2.6, txt, transform=cb.ax.transAxes, ha='center', va='top', fontsize=6.5)
        return im
    cb.set_label(lab, fontsize=7.5, labelpad=2)
    if mode == 'twotone':
        for y, txt in ((1.035, 'polymer-rich'), (-0.035, 'crowder-rich')):
            cb.ax.text(0.0, y, txt, transform=cb.ax.transAxes, ha='left', va='bottom' if y > 0 else 'top', fontsize=6.5)
    return im
