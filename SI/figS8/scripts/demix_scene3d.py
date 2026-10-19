"""Block d of fig:polymer, 3-D companion of each slice (2026-10-09): the whole 64^3 box at the slice's time, polymer and
crowders as points, with the slab that the slice averages outlined.

WHAT IS DRAWN
    Polymer: free monomers at t = 0 (the .lm initial lattice, species M; no chain exists yet), chain subunits after
    (snap_chain, voxel indices). Crowders: the finite-size bodies' footprint voxels (snap_body) or, in the point arm, the
    point crowders per voxel (snap_bath). One point per occupied voxel (per particle in a voxel holding several), at the
    voxel centre: the lattice resolves nothing finer. A random subset of at most NPTS per species is drawn (fixed seed,
    the same for every panel), because tens of thousands of points in an inch-wide box print as solid colour.

ORIENTATION
    The same as the slice beside it: up is x (the polymer starts in x >= L/2, the top), across is z, and depth is y, the
    axis the slice averages over. The outlined rectangle is the middle of y, where the slice is taken.

Orthographic projection, points sorted back to front and faded toward white with depth so the near face reads first.
"""
import os

import h5py
import numpy as np

import fig_polymer_r2 as F

NPTS = 2600                     # points per species per panel
AZ, EL = -58.0, 20.0            # camera azimuth / elevation, degrees
FADE = 0.55                     # far points are mixed this far toward white
OUTLINE = '#2bb3a3'             # the slice plane, the frame of the slice beside it and the rules between (fig:rxthree's)


def _basis(az, el):
    a, e = np.radians(az), np.radians(el)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return right, up, toward


def _world(x, y, z):
    """Lattice (x, y, z) -> world (across, depth, up) = (z, y, x)."""
    return np.stack([z, y, x], axis=-1).astype(float)


def _unravel(v, nx):
    v = np.asarray(v, np.int64)                                   # v = z*nx*nx + y*nx + x
    return v % nx, (v // nx) % nx, v // (nx * nx)


def points(root, run, t):
    """(poly_xyz, crowd_xyz, nx, t) in world coordinates, voxel centres, at the snapshot nearest t."""
    import demix_slab_maps as DS
    d = np.load(os.path.join(root, run + '.npz'), allow_pickle=True)
    nx = int(d['arg_nx'])
    k = DS._snap(d, t)
    if float(d['snap_t_%d' % k]) == 0.0:
        L = h5py.File(os.path.join(root, run + '.lm'), 'r')['Simulations/0000001/Lattice/0000000000'][:]
        cnt = (L == 1).sum(axis=3)                                # [x, y, z]
        idx = np.argwhere(cnt > 0)
        rep = cnt[tuple(idx.T)]
        x, y, z = np.repeat(idx, rep, axis=0).T
    else:
        x, y, z = _unravel(d['snap_chain_%d' % k], nx)
    poly = _world(x, y, z) + 0.5
    bath = 'snap_bath_%d' % k
    if bath in d.files:
        n = np.asarray(d[bath], np.int64)
        v = np.repeat(np.nonzero(n)[0], n[n > 0])
    else:
        v = d['snap_body_%d' % k]
    cx, cy, cz = _unravel(v, nx)
    return poly, _world(cx, cy, cz) + 0.5, nx, float(d['snap_t_%d' % k])


def _proj(p, B):
    right, up, toward = B
    return p @ right, p @ up, p @ toward


def draw_box(ax, run_pts, nx, color_poly, color_crowd, seed=7):
    """One panel: box edges, the subsampled points back to front, the slice outline, the near edges."""
    import matplotlib.colors as mc
    B = _basis(AZ, EL)
    rng = np.random.default_rng(seed)
    poly, crowd = run_pts
    sel = []
    for P, col in ((poly, color_poly), (crowd, color_crowd)):
        if len(P) > NPTS:
            P = P[rng.choice(len(P), NPTS, replace=False)]
        sel.append((P, np.tile(mc.to_rgb(col), (len(P), 1))))
    P = np.concatenate([s[0] for s in sel]); C = np.concatenate([s[1] for s in sel])
    u, w, dep = _proj(P, B)
    o = np.argsort(dep)                                           # far first
    dn = (dep - dep.min()) / max(np.ptp(dep), 1e-9)
    C = C + (1.0 - C) * (FADE * (1.0 - dn))[:, None]

    corners = np.array([[i, j, k] for i in (0, nx) for j in (0, nx) for k in (0, nx)], float)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.sum(corners[a] != corners[b]) == 1]
    cu, cw, cd = _proj(corners, B)
    near = int(np.argmax(cd))
    for a, b in edges:                                            # far edges under the points
        if near not in (a, b):
            ax.plot([cu[a], cu[b]], [cw[a], cw[b]], color=F.KP.CAGE, lw=0.45, zorder=1)
    ax.scatter(u[o], w[o], s=0.55, c=C[o], marker='o', linewidths=0, zorder=2, rasterized=True)
    h = nx / 2.0                                                  # the slice plane: middle of the depth axis
    pl = np.array([[0, h, 0], [nx, h, 0], [nx, h, nx], [0, h, nx], [0, h, 0]], float)
    pu, pw, _ = _proj(pl, B)
    ax.plot(pu, pw, color=OUTLINE, lw=1.4, zorder=5, solid_joinstyle='miter')
    for a, b in edges:                                            # the three edges at the near corner on top
        if near in (a, b):
            ax.plot([cu[a], cu[b]], [cw[a], cw[b]], color=F.INK, lw=0.5, zorder=4)
    ax.set_aspect('equal'); ax.axis('off')
    pad = 0.02 * nx
    ax.set_xlim(cu.min() - pad, cu.max() + pad); ax.set_ylim(cw.min() - pad, cw.max() + pad)
    return np.stack([pu[:4], pw[:4]], axis=1)


def panel_boxes(axes, root, arms, times, colors):
    """axes[row][col]: rows = arms ((run, label), ...), columns = times. colors = (polymer, crowder). Returns the
    projected slice-plane corners per panel."""
    planes = [[None] * len(times) for _ in arms]
    for r, (run, name) in enumerate(arms):
        for c, t in enumerate(times):
            poly, crowd, nx, tt = points(root, run, t)
            planes[r][c] = draw_box(axes[r][c], (poly, crowd), nx, colors[0], colors[1])
            print('  %-16s t=%4.1f  3-D: %d polymer, %d crowder points (drawn <= %d each)'
                  % (name, tt, len(poly), len(crowd), NPTS))
    return planes


def snapshots(fig, AX, YT, textw, root, arms, colors, Y0, W=0.86, X0=0.40, times=(0, 1, 10)):
    """fig:phase c: for each time the 3-D box, then its slice (demix_slab_maps), ruled together, the colour bar standing
    at the right. colors = (polymer, crowder). AX/YT: the figure's inch layout helpers. Returns the bottom (in)."""
    import demix_slab_maps as DS
    from matplotlib.patches import ConnectionPatch
    G, P = 0.04, 0.10                                             # P: the gap between one time's pair and the next
    XS = [X0 + c * (2 * W + G + P) for c in range(len(times))]
    rows = range(len(arms))
    boxes = [[fig.add_axes(AX(x, Y0 + r * (W + G), W, W)) for x in XS] for r in rows]
    planes = panel_boxes(boxes, root, arms, times, colors)
    grid = [[fig.add_axes(AX(x + W + G, Y0 + r * (W + G), W, W)) for x in XS] for r in rows]
    H = len(arms) * (W + G) - G
    cax = fig.add_axes(AX(XS[-1] + 2 * W + G + 0.10, Y0 + 0.12, 0.06, H - 0.24))
    DS.panel_maps(grid, cax, root, arms, times, 'twotone', titles=False, labels=False, frame=OUTLINE, frame_lw=1.4)
    for r in rows:                     # rule the plane's two right-hand corners to the slice's left-hand ones (fig:rxthree d)
        for c in range(len(times)):
            pl = planes[r][c]
            right = np.argsort(pl[:, 0])[-2:]
            top, bot = right[np.argsort(pl[right, 1])[::-1]]
            for src, dst in ((top, (0.0, 1.0)), (bot, (0.0, 0.0))):
                fig.add_artist(ConnectionPatch(xyA=tuple(pl[src]), coordsA=boxes[r][c].transData, xyB=dst,
                                               coordsB=grid[r][c].transAxes, color=OUTLINE, lw=0.9, zorder=0))
    for x, t in zip(XS, times):
        fig.text((x + W + G / 2) / textw, YT(Y0 - 0.04), r'$t = %d$ s' % t, ha='center', va='bottom', fontsize=8)
    for r, (_, name) in enumerate(arms):
        fig.text((X0 - 0.06) / textw, YT(Y0 + r * (W + G) + W / 2), name, rotation=90, ha='right', va='center',
                 fontsize=7.5, color=F.INK)
    return Y0 + H


def lengths(ax_t, ax_h, root, arms, colors):
    """fig:phase b, kept small: chain length over time and its distribution at t = 10 s (demix_lengths).
    colors = (polymer, crowder) as for snapshots; the lines take the arm colours (finite-size = crowder blue)."""
    import demix_lengths as DL
    DL.panel_lengths(ax_t, ax_h, arms, root, colors[::-1], ('-', '--'), note=False)
    ax_h.set_xlabel('length at $t = 10$ s'); ax_t.set_ylabel('chain length')
    for ax in (ax_t, ax_h):
        ax.tick_params(labelsize=6.5); ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    ax_h.set_xticks([0, 30, 60])
    ax_h.legend(loc='upper right', fontsize=5.5, frameon=False, handlelength=1.1, borderpad=0.1, labelspacing=0.15,
                handletextpad=0.4)


# ---- the publication page (fig:phase, 2026-10-19; FIGURE_STYLE.md) --------------------------------------------------
def snapshots_pub(fig, root, arms, colors, x0, y0, W=1.38, G=0.04, P=0.26, GR=0.10, label_w=0.62, times=(0, 1, 10)):
    """fig:phase c on the 8.5 x 11 page, TRANSPOSED against snapshots(): one row per time (t = ... at the left), one
    pair of columns per arm (the 3-D box, then its slice, the arm named above the pair), the colour bar at the right.
    x0, y0, W, G (box-slice gap), P (gap between arms), GR (gap between rows): inches from the page's top-left.
    Returns (bottom, right) in inches."""
    import jcp_style as J
    import demix_slab_maps as DS
    from matplotlib.patches import ConnectionPatch
    X0 = x0 + label_w
    XA = [X0 + r * (2 * W + G + P) for r in range(len(arms))]
    YR = [y0 + c * (W + GR) for c in range(len(times))]
    boxes = [[J.pub_ax(fig, XA[r], YR[c], W, W) for c in range(len(times))] for r in range(len(arms))]
    planes = panel_boxes(boxes, root, arms, times, colors)
    grid = [[J.pub_ax(fig, XA[r] + W + G, YR[c], W, W) for c in range(len(times))] for r in range(len(arms))]
    H = len(times) * (W + GR) - GR
    xr = XA[-1] + 2 * W + G
    cax = J.pub_ax(fig, xr + 0.12, y0 + 0.30, 0.12, H - 0.60)
    DS.panel_maps(grid, cax, root, arms, times, 'twotone', titles=False, labels=False, frame=OUTLINE, frame_lw=1.4,
                  pub=True)
    for r in range(len(arms)):
        for c in range(len(times)):
            pl = planes[r][c]
            right = np.argsort(pl[:, 0])[-2:]
            top, bot = right[np.argsort(pl[right, 1])[::-1]]
            for src, dst in ((top, (0.0, 1.0)), (bot, (0.0, 0.0))):
                fig.add_artist(ConnectionPatch(xyA=tuple(pl[src]), coordsA=boxes[r][c].transData, xyB=dst,
                                               coordsB=grid[r][c].transAxes, color=OUTLINE, lw=0.9, zorder=0))
    tx = lambda x, y, s, **kw: fig.text(x / J.PAGE_W, 1.0 - y / J.PAGE_H, s, **kw)
    for x, (_, name) in zip(XA, arms):
        tx(x + W + G / 2, y0 - 0.08, name, ha='center', va='bottom', fontsize=J.LABEL_PT, color=F.INK)
    for y, t in zip(YR, times):
        tx(X0 - 0.10, y + W / 2, r'$t$ = %d s' % t, ha='right', va='center', fontsize=J.LABEL_PT, color=F.INK)
    return y0 + H, xr


def lengths_pub(ax_t, ax_h, root, arms, colors):
    """fig:phase b on the publication page: as lengths(), with the house 10/11 pt type."""
    import jcp_style as J
    import demix_lengths as DL
    DL.panel_lengths(ax_t, ax_h, arms, root, colors[::-1], ('-', '--'), note=False)
    ax_h.set_xlabel('chain length at $t$ = 10 s'); ax_t.set_ylabel('chain length')
    ax_h.set_xticks([0, 20, 40, 60]); ax_h.set_yticks([0, 0.04, 0.08]); ax_h.set_yticklabels(['0', '0.04', '0.08'])
    ax_h.legend(loc='upper right', fontsize=J.TEXT_PT, frameon=False, handlelength=1.8, borderpad=0.2)
