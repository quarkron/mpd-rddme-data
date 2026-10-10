"""Three-dimensional snapshots of the ternary relaxation lattice: the 80 nm cell, the bath it holds,
and the 20 nm body sitting in it.

    python rxthree_scene3d.py --run mpd_crddme/relax3_out/lat3_model_s0.npz --out out/scene3d

WHAT IS DRAWN, AND WHAT IS LITERAL
    The simulation is a 320 nm box of 32^3 voxels carrying 64 copies of the same 80 nm cell, so one
    cell -- eight voxels on a side, with one body at its centre-right -- is the unit worth drawing.
    Every voxel of that cell is drawn as a cube, coloured and made opaque by the volume fraction the
    bath actually occupies there,

        eta(v) = (pi/6) [ n_A(v) sigma_A^3 + n_B(v) sigma_B^3 ] / h^3,

    the same eta the manuscript's Eq. for the composition moments defines, evaluated per subvolume.
    That is the lattice's own state variable rather than a rendering of particles it does not track:
    the model resolves occupancy per voxel and nothing finer, and drawing individual spheres at
    invented sub-voxel positions would claim resolution the method does not have. Depth accumulation
    does the rest: eight layers at the bulk density build to about two thirds opacity, so a filled
    half reads as solid and an empty half reads as clear.

    CUBES, NOT BILLBOARDS. Each voxel contributes the three of its faces that point toward the
    camera, so the cell reads as a stack of cells rather than as a fog of squares. The facets are
    Lambertian shaded from the same light as the body, but only between 0.75 and 1.0 of base rather
    than the usual full range: shading is there to give the cubes their edges, and taking a facet
    down to half brightness would move its colour a quarter of the way along the ramp and break the
    reading against the colour bar.

    The ramp is the palette's white-to-navy occupancy ramp, not its full spectral one. Density is a
    single ordered quantity, so it reads correctly only on a monotone ramp; and on a white page the
    spectral ramp ends pale, which would leave a dense voxel and an empty one both light.

    The body is the one object drawn as itself, a 20 nm sphere at its true position in the cell,
    because that is what the structure lattice carries.

A CUBE IS CONTEXT, A SLICE IS EVIDENCE
    Eight layers of translucent cubes show where the density is but not what it is: every value the
    reader might want to compare against the colour bar has seven other layers in front of it. So the
    cube is drawn small, as the map, and one layer of it -- the one the body's centre sits in -- is
    outlined and enlarged beside it, where each subvolume is a flat square at its own colour with
    nothing over it. The slice and the plane it came from carry the same colour, and the pair is
    ruled corner to corner. The enlargement keeps the cube's orientation: with the default view both
    put the slab axis across the page and the second axis up it.

PROJECTION
    Orthographic, azimuth and elevation from --view. Every quad, line and sphere in a scene is depth
    sorted into one zorder scale, so the near cage passes in front of the bath and the far cage
    behind it, and the bath in front of the body veils it exactly as far as its density warrants.
    Lambertian shading from one light fixed in camera space, so both scenes are lit identically.

Runs in the lm2.6 container (numpy + matplotlib); palette from the Kip3 traffic figures.
"""
import argparse, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))
import kip3_palette as P

CELL = 8                      # voxels per 80 nm cell; the campaign's slab period
SIG = {'A': 2.0, 'B': 4.0, 'C': 20.0}
H = 10.0                      # nm per voxel
ETA_MAX = 0.30                # the slab starts at twice the bulk 0.14; this is the colour ceiling
ZERO = np.zeros(3)
BODY = (0.85, 0.45, 0.13)     # the body, warm against the cool bath ramp
HILITE = '#2bb3a3'            # the cut plane and the frame of its enlargement, one colour for the pair


def basis(az_deg, el_deg):
    """Orthographic camera basis: (right, up, toward-camera) unit vectors."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    c = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    r = np.array([np.cos(az), np.sin(az), 0.0])
    u = np.cross(c, r)
    return r, u / np.linalg.norm(u), c


def project(pts, B, off=ZERO):
    """(..., 3) world -> (..., 2) screen and (...) depth; larger depth is nearer the camera."""
    r, u, c = B
    q = np.asarray(pts, float) + np.asarray(off, float)
    return np.stack([q @ r, q @ u], -1), q @ c


def mix(rgb, other, t):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(rgb)), np.array(to_rgb(other))
    return tuple(np.clip((1 - t) * a + t * b, 0, 1))


def zscaler(dmin, dmax):
    span = max(dmax - dmin, 1e-9)
    return lambda d: 1.0 + 998.0 * (d - dmin) / span


def eta_of(nA, nB):
    """Bath volume fraction per voxel from the two occupancies."""
    return (np.pi / 6.0) * (nA * SIG['A'] ** 3 + nB * SIG['B'] ** 3) / H ** 3


def draw_box(ax, lo, hi, B, zscale, off=ZERO, grid=True):
    """The cell boundary, and one faint line per voxel on the three far faces so the scene reads as
    a lattice rather than as a fog in a crate."""
    from matplotlib.lines import Line2D
    _, _, c = B
    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        xy, dep = project(corners[[i, j]], B, off)
        ax.add_line(Line2D(xy[:, 0], xy[:, 1], color=P.CAGE, lw=0.45, alpha=0.9,
                           zorder=zscale(dep.mean()), solid_capstyle='round'))
    if not grid:
        return
    # Only the faces pointing away from the camera are ruled: ruling the near ones would lay a grid
    # over the bath the reader is meant to see through.
    for axis in range(3):
        nrm = np.zeros(3); nrm[axis] = 1.0
        for sgn, plane in ((+1.0, hi), (-1.0, lo)):
            if float(np.dot(sgn * nrm, c)) > 0.0:
                continue                                   # this face is toward the camera
            for t in np.arange(lo, hi + 1e-9, 1.0):
                for other in range(3):
                    if other == axis:
                        continue
                    third = 3 - axis - other
                    p0 = np.zeros(3); p1 = np.zeros(3)
                    p0[axis] = p1[axis] = plane
                    p0[other] = p1[other] = t
                    p0[third], p1[third] = lo, hi
                    xy, dep = project(np.array([p0, p1]), B, off)
                    ax.add_line(Line2D(xy[:, 0], xy[:, 1], color=P.CAGE, lw=0.2, alpha=0.40,
                                       zorder=zscale(dep.mean() - 0.02), solid_capstyle='butt'))


def shade(base_rgb, normal, light, ambient=0.75):
    """Lambertian, but floored high: see CUBES, NOT BILLBOARDS above."""
    lam = max(0.0, float(np.dot(normal / np.linalg.norm(normal), light)))
    return tuple(np.clip(np.asarray(base_rgb[:3]) * (ambient + (1.0 - ambient) * lam), 0, 1))


def draw_bath(ax, eta, B, zscale, cmap, light, off=ZERO, alpha_max=0.145):
    """One cube per occupied voxel -- its three camera-facing faces -- coloured and made opaque by eta.

    alpha_max is below what the single-square version used, because a voxel now lays down three
    facets instead of one; not a third of it, though, since the three tile adjacent regions of the
    screen rather than stacking, so the accumulated opacity through the cell is nearer twice than
    three times. It was set by matching the filled half against the square-splat render."""
    from matplotlib.patches import Polygon
    _, _, c = B
    idx = np.argwhere(eta > 1e-9)
    if not len(idx):
        return
    centres = idx.astype(float) + 0.5
    _, dep = project(centres, B, off)
    order = np.argsort(dep)                                 # far first, so near cubes land on top
    for k in order:
        lo = idx[k].astype(float)
        f = min(float(eta[tuple(idx[k])]) / ETA_MAX, 1.0)
        base = cmap(0.15 + 0.85 * f)
        a = alpha_max * (0.30 + 0.70 * f)
        z = zscale(dep[k])
        for axis in range(3):
            nrm = np.zeros(3); nrm[axis] = 1.0
            s = 1.0 if float(np.dot(nrm, c)) > 0.0 else -1.0
            o1, o2 = [q for q in range(3) if q != axis]
            quad = np.zeros((4, 3))
            for m, (u1, u2) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                quad[m, axis] = lo[axis] + (1.0 if s > 0 else 0.0)
                quad[m, o1] = lo[o1] + u1
                quad[m, o2] = lo[o2] + u2
            xy, _ = project(quad, B, off)
            ax.add_patch(Polygon(xy, closed=True, facecolor=shade(base, s * nrm, light),
                                 edgecolor='none', alpha=a, zorder=z))


def draw_sphere(ax, centre, radius, rgb, B, zscale, light, off=ZERO):
    """Three nested vector circles stepped toward the light; vector, so the PDF stays small."""
    from matplotlib.patches import Circle
    r_, u_, _ = B
    xy, dep = project(np.asarray(centre, float), B, off)
    z = zscale(dep)
    lx, ly = float(np.dot(light, r_)), float(np.dot(light, u_))
    nrm = np.hypot(lx, ly) or 1.0
    lx, ly = lx / nrm, ly / nrm
    for k, (fr, o, tint) in enumerate([(1.00, 0.00, 0.00), (0.70, 0.22, 0.26), (0.38, 0.36, 0.55)]):
        ax.add_patch(Circle((xy[0] + o * radius * lx, xy[1] + o * radius * ly), fr * radius,
                            facecolor=mix(rgb, '#ffffff', tint),
                            edgecolor=mix(rgb, '#000000', 0.45) if k == 0 else 'none',
                            lw=0.3 if k == 0 else 0.0, zorder=z + 0.001 * (k + 1)))


def plane_corners(k, axis, cell=CELL):
    """The four corners of the mid-plane of voxel layer `k` normal to `axis`, in cell coordinates."""
    q = np.zeros((4, 3))
    o1, o2 = [a for a in range(3) if a != axis]
    for m, (u1, u2) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        q[m, axis] = k + 0.5
        q[m, o1] = u1 * cell
        q[m, o2] = u2 * cell
    return q


def draw_slice_plane(ax, k, axis, B, off=ZERO, cell=CELL, color=HILITE, lw=1.1):
    """Outline the layer that the blown-up slice is cut from, drawn in front of everything so it is
    visible through the bath. Returns the projected corners, in the same order as plane_corners, so
    the caller can rule across to the enlargement."""
    from matplotlib.patches import Polygon
    q = plane_corners(k, axis, cell)
    xy, _ = project(q, B, off)
    ax.add_patch(Polygon(xy, closed=True, facecolor='none', edgecolor=color, lw=lw, zorder=1001))
    return xy


def draw_slice2d(ax, eta, bodies, k, axis, cmap, cell=CELL, color=HILITE):
    """The blown-up slice: one square per subvolume of layer `k`, on the same colour scale as the
    cubes, with the subvolume edges drawn so the lattice is countable and the body drawn as the disc
    its sphere actually cuts on this plane.

    The radius is the CHORD, sqrt(R^2 - d^2) for a centre d from the plane, not R: a sphere whose
    centre sits off the plane meets it in a smaller circle, and drawing it at full size would put
    more of the body in the slice than is there."""
    from matplotlib.patches import Circle
    o1, o2 = [a for a in range(3) if a != axis]
    img = np.take(eta, k, axis=axis)                    # (o1, o2)
    ax.imshow(img.T, origin='lower', extent=[0, cell, 0, cell], cmap=cmap, vmin=0.0, vmax=ETA_MAX,
              interpolation='nearest', zorder=0)
    for t in range(cell + 1):
        ax.axvline(t, color=P.CAGE, lw=0.3, alpha=0.55, zorder=1)
        ax.axhline(t, color=P.CAGE, lw=0.3, alpha=0.55, zorder=1)
    R = 0.5 * SIG['C'] / H
    for c in bodies:
        d = abs(float(c[axis]) - k)
        if d >= R:
            continue
        ax.add_patch(Circle((c[o1] + 0.5, c[o2] + 0.5), math.sqrt(R * R - d * d),
                            facecolor=BODY, edgecolor=mix(BODY, '#000000', 0.45), lw=0.5, zorder=3))
    ax.set_xlim(0, cell); ax.set_ylim(0, cell); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.1); sp.set_color(color)


def cell_of(d, fi, origin=(0, 0, 0), cell=CELL):
    """The eta field of one 80 nm cell at frame `fi`, and the bodies inside it, in cell coordinates."""
    import json
    meta = json.loads(str(d['meta']))
    N = int(meta['N'])
    o = np.asarray(origin, int)
    sl = tuple(slice(o[a], o[a] + cell) for a in range(3))
    A = d['occA'][fi].reshape(N, N, N)[sl].astype(float)
    B_ = d['occB'][fi].reshape(N, N, N)[sl].astype(float)
    pc = np.asarray(d['posC'][fi], float) - o
    inside = np.all((pc >= -0.5) & (pc < cell - 0.5), axis=1)
    return eta_of(A, B_), pc[inside]


def draw_scene(ax, eta, bodies, B, light, off=ZERO, cmap=None, cell=CELL, grid=True):
    """One cell: the far cage, the bath, the bodies, the near cage, depth sorted into one scale."""
    cmap = cmap or P.cmap_binary()
    corners = np.array([[x, y, z] for x in (0, cell) for y in (0, cell) for z in (0, cell)])
    _, dep = project(corners, B, off)
    zscale = zscaler(dep.min() - 0.5, dep.max() + 0.5)
    draw_box(ax, 0.0, float(cell), B, zscale, off, grid=grid)
    draw_bath(ax, eta, B, zscale, cmap, light, off)
    for c in bodies:
        draw_sphere(ax, c + 0.5, 0.5 * SIG['C'] / H, BODY, B, zscale, light, off)
    xy, _ = project(corners, B, off)
    return xy


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--frames', default='0,-1', help='frame indices to draw, comma separated')
    p.add_argument('--view', default='28,18')
    p.add_argument('--out', required=True)
    A = p.parse_args()

    d = np.load(A.run, allow_pickle=True)
    fr = [int(x) % len(d['tau']) for x in A.frames.split(',')]
    az, el = (float(x) for x in A.view.split(','))
    B = basis(az, el)
    light = np.array([0.35, -0.55, 0.76]); light /= np.linalg.norm(light)

    fig, axs = plt.subplots(1, len(fr), figsize=(3.0 * len(fr), 3.0))
    for ax, fi in zip(np.atleast_1d(axs), fr):
        eta, bodies = cell_of(d, fi)
        xy = draw_scene(ax, eta, bodies, B, light)
        ax.set_xlim(xy[:, 0].min() - 0.6, xy[:, 0].max() + 0.6)
        ax.set_ylim(xy[:, 1].min() - 0.6, xy[:, 1].max() + 0.6)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(r'$\tau = %.3g$' % d['tau'][fi])
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), dpi=300, bbox_inches='tight')
    print('wrote %s.pdf/.png' % A.out)


if __name__ == '__main__':
    main()
