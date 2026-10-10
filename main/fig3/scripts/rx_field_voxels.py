"""The lattice box drawn as a solid of shaded voxel cells, for the configuration block.

    draw_box(ax, cells, basis(24, 14), light, cmap, vmin, vmax)

WHAT IS LITERAL, AND WHAT IS NOT
    The cubes are the simulation's own 10 nm voxels at their true cubic aspect, and the run of
    them drawn is a real run of the lattice -- two periods of the field, sixteen voxels, one
    stack deep. What is not literal is the shading: a cube carries the mean occupancy of the
    plane of voxels normal to the field that it belongs to, not its own count. That is forced by
    counting statistics rather than chosen. A voxel holds about three particles, so its Poisson
    fluctuation is 70 % of its mean while the field response is 17 %: a stack shaded from raw
    per-voxel counts is visually pure noise, measured before it was believed. The plane averages
    a thousand voxels and fluctuates by 2 %, so the stack shows the profile and nothing else.

    Only the three faces of the solid that point toward the camera are drawn; it is opaque, so
    there is nothing behind them to see. The fine ruling is one line per voxel along the field
    axis, which is what makes the run of sixteen countable.

Palette and projection follow rxthree_scene3d.py, whose orthographic camera this shares.
"""
import numpy as np

ZERO = np.zeros(3)


def basis(az_deg, el_deg):
    """Orthographic camera basis: (right, up, toward-camera) unit vectors."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    c = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    r = np.array([np.cos(az), np.sin(az), 0.0])
    u = np.cross(c, r)
    return r, u / np.linalg.norm(u), c


def project(pts, B, off=ZERO):
    """(..., 3) world -> (..., 2) screen."""
    r, u, _ = B
    q = np.asarray(pts, float) + np.asarray(off, float)
    return np.stack([q @ r, q @ u], -1)


def shade(rgb, normal, light, ambient=0.72):
    """Lambertian, floored high: the shading is there to give the faces their edges, and taking
    one down to half brightness would move its colour a quarter of the way along the ramp and
    break the reading against the colour bar."""
    lam = max(0.0, float(np.dot(normal, light)))
    return tuple(np.clip(np.asarray(rgb[:3]) * (ambient + (1.0 - ambient) * lam), 0, 1))


def _face_quads(shape, size, axis, near):
    """Corner grids of every cell face on one outer face of the solid, as (n, 4, 3) world points."""
    o1, o2 = [q for q in range(3) if q != axis]
    plane = shape[axis] * size[axis] if near else 0.0
    quads = []
    for a in range(shape[o1]):
        for b in range(shape[o2]):
            q = np.zeros((4, 3))
            for m, (u1, u2) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                q[m, axis] = plane
                q[m, o1] = (a + u1) * size[o1]
                q[m, o2] = (b + u2) * size[o2]
            quads.append((q, a, b))
    return quads


def draw_box(ax, cells, B, light, cmap, vmin, vmax, off=ZERO, size=None, rule=None, cage='#93aac2'):
    """Draw the solid and return the projected corners of its bounding box.

    `cells` is (nx, ny, nz) mean occupancy; `size` the world extent of one cell, defaulting to a
    cube of side 1 along the field axis and shape[0]/shape[i] across it, which makes the drawn
    solid cubic whatever the pooling -- pass ones for a stack of true cubes instead. `rule`
    divides the field axis into that many fine lines.
    """
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    cells = np.asarray(cells, float)
    shape = cells.shape
    size = size if size is not None else \
        np.array([1.0, shape[0] / float(shape[1]), shape[0] / float(shape[2])])
    span = np.array(shape) * size
    _, _, cam = B
    norm = (lambda v: np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0, 1))

    for axis in range(3):
        nrm = np.zeros(3)
        nrm[axis] = 1.0
        near = float(np.dot(nrm, cam)) > 0.0          # this outer face points at the camera
        s = nrm if near else -nrm
        o1, o2 = [q for q in range(3) if q != axis]
        for quad, a, b in _face_quads(shape, size, axis, near):
            idx = [0, 0, 0]
            idx[axis] = shape[axis] - 1 if near else 0
            idx[o1], idx[o2] = a, b
            rgb = cmap(norm(cells[tuple(idx)]))
            ax.add_patch(Polygon(project(quad, B, off), closed=True, lw=0,
                                 facecolor=shade(rgb, s, light), edgecolor='none', zorder=2))

    # Fine ruling along the field axis, on the two visible faces that carry it, plus the outline
    # of the three visible faces. The solid is opaque, so its far edges are not drawn at all.
    for axis in range(3):
        nrm = np.zeros(3)
        nrm[axis] = 1.0
        near = float(np.dot(nrm, cam)) > 0.0
        plane = span[axis] if near else 0.0
        o1, o2 = [q for q in range(3) if q != axis]
        corner = np.zeros((5, 3))
        for m, (u1, u2) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))):
            corner[m, axis] = plane
            corner[m, o1] = u1 * span[o1]
            corner[m, o2] = u2 * span[o2]
        xy = project(corner, B, off)
        ax.add_line(Line2D(xy[:, 0], xy[:, 1], color=cage, lw=0.55, alpha=0.95, zorder=4,
                           solid_capstyle='round'))
        if rule and axis != 0:                        # faces that show the field axis
            for t in np.linspace(0.0, span[0], rule + 1):
                p = np.zeros((2, 3))
                p[:, axis] = plane
                p[:, 0] = t
                other = o1 if o1 != 0 else o2
                p[0, other], p[1, other] = 0.0, span[other]
                xy = project(p, B, off)
                ax.add_line(Line2D(xy[:, 0], xy[:, 1], color=cage, lw=0.14, alpha=0.35,
                                   zorder=3, solid_capstyle='butt'))

    corners = np.array([[x, y, z] for x in (0, span[0]) for y in (0, span[1])
                        for z in (0, span[2])])
    return project(corners, B, off)
