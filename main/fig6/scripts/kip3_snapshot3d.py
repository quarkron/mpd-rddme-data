"""Three-dimensional snapshots of the Kip3 traffic lattice: the tube, its voxel cage, the motors.

    python kip3_snapshot3d.py --run out/fig1c_exc_iso_1500.npz --mode time \
        --times 100,250,400,1200 --out out/fig_snap3d
    python kip3_snapshot3d.py --run out/fig1c_exc_iso_1500.npz --mode space \
        --time 1200 --out out/fig_snap3d_space

MODES
    time   one window near the plus end at several times, the slices staggered along the depth
           axis so the sequence recedes: earliest at the back and top, latest nearest the reader.
           The jam front sweeps through the window, so the slices show it filling.
    space  three windows along the tube at one instant - the dilute antenna region, the domain
           wall (located from the density profile, not assumed), and the interior of the jam.

WHAT IS DRAWN, AND WHAT IS LITERAL
    The microtubule is drawn as a shaded cylinder because that is the object the model stands for:
    a 25-nm tube, inscribed here in the 3x3 block of voxels the lattice actually reserves for it.
    The 16 shell voxels around that block are drawn literally, as the wire cage, one cell per
    binding site per lane. Motors are beads at the centres of the shell voxels they occupy: one
    bead is one particle in one voxel at its true lattice position, drawn at 0.84 h so the beads
    read as separate objects (their hard-sphere diameter is 0.999 h, which would have neighbours
    touching).

    Beside each slice is the whole 10-um microtubule at that instant - the lane-averaged density
    along its full length - with the rendered window boxed on it and ruled across to the render,
    so the magnified view is placed in the track it was cut from.

PROJECTION
    Orthographic, azimuth and elevation from --view. Every element in a panel - cylinder quads,
    cage lines, beads, and across every time slice - is depth sorted into one zorder scale, so
    the cage passes in front of the near motors and behind the far ones, a slice occludes the one
    behind it, and the tube hides what is behind it. Lambertian shading from one light fixed in
    camera space, so every panel and slice is lit identically.

Runs in the lm2.6 container (numpy + matplotlib); jcp_style from studies/common.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M
import kip3_palette as P
from kip3_analyze import load, kymo, style, SHORT, _key

BODY_LO, BODY_HI = 3, 5          # body voxels span [3, 5] in y and z
SHELL_LO, SHELL_HI = 2, 6        # shell ring spans [2, 6]; centre is 4
CENTRE = 0.5 * (SHELL_LO + SHELL_HI)
ZERO = np.zeros(3)


def basis(az_deg, el_deg):
    """Orthographic camera basis: (right, up, toward-camera) unit vectors. The tube axis is x, so
    a small azimuth keeps the tube nearly horizontal on the page."""
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


def shade(base_rgb, normal, light, ambient=0.42):
    lam = max(0.0, float(np.dot(normal / np.linalg.norm(normal), light)))
    return tuple(np.clip(np.asarray(base_rgb) * (ambient + (1.0 - ambient) * lam), 0, 1))


def mix(rgb, other, t):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(rgb)), np.array(to_rgb(other))
    return tuple(np.clip((1 - t) * a + t * b, 0, 1))


def perimeter_path():
    """Vertices of the shell's outer boundary in (y, z): the 5x5 square's perimeter stepped one
    voxel at a time, so every cage line sits on a real cell edge."""
    a, b = SHELL_LO - 0.5, SHELL_HI + 0.5
    n = int(round(b - a))
    pts = ([(a + i, a) for i in range(n)] + [(b, a + i) for i in range(n)] +
           [(b - i, b) for i in range(n)] + [(a, b - i) for i in range(n)])
    return pts + [pts[0]]


def draw_cylinder(ax, x0, x1, R, B, light, zscale, off=ZERO, n_theta=44, n_x=10):
    """The microtubule, with a faint 13-fold shading modulation for the protofilaments."""
    from matplotlib.patches import Polygon
    th = np.linspace(0, 2 * np.pi, n_theta + 1)
    xs = np.linspace(x0, x1, n_x + 1)
    _, _, c = B
    for i in range(n_theta):
        t0, t1 = th[i], th[i + 1]
        tm = 0.5 * (t0 + t1)
        nrm = np.array([0.0, np.cos(tm), np.sin(tm)])
        if float(np.dot(nrm, c)) <= 0.0:
            continue
        col = shade(np.asarray(P.TUBE) * (0.93 + 0.07 * np.cos(13.0 * tm)), nrm, light)
        for j in range(n_x):
            quad = np.array([[xs[j], CENTRE + R * np.cos(t0), CENTRE + R * np.sin(t0)],
                             [xs[j + 1], CENTRE + R * np.cos(t0), CENTRE + R * np.sin(t0)],
                             [xs[j + 1], CENTRE + R * np.cos(t1), CENTRE + R * np.sin(t1)],
                             [xs[j], CENTRE + R * np.cos(t1), CENTRE + R * np.sin(t1)]])
            xy, dep = project(quad, B, off)
            ax.add_patch(Polygon(xy, closed=True, facecolor=col, edgecolor=col, lw=0.15,
                                 zorder=zscale(dep.mean())))


def draw_cage(ax, s0, s1, B, zscale, off=ZERO, every=1):
    from matplotlib.lines import Line2D
    per = perimeter_path()
    for (y, z) in per[:-1]:
        xy, dep = project(np.array([[s0 - 0.5, y, z], [s1 + 0.5, y, z]]), B, off)
        ax.add_line(Line2D(xy[:, 0], xy[:, 1], color=P.CAGE, lw=0.3, alpha=0.75,
                           zorder=zscale(dep.mean()), solid_capstyle='butt'))
    for s in range(s0, s1 + 2, every):
        xy, dep = project(np.array([[s - 0.5, y, z] for (y, z) in per]), B, off)
        ax.add_line(Line2D(xy[:, 0], xy[:, 1], color=P.CAGE, lw=0.22, alpha=0.55,
                           zorder=zscale(dep.mean())))


def draw_sphere(ax, centre, radius, rgb, B, zscale, light, off=ZERO):
    """A bead: three nested vector circles stepped toward the light. Vector, so the PDF stays
    small and scales; at 5-10 pt on the page it reads as a shaded sphere."""
    from matplotlib.patches import Circle
    r_, u_, _ = B
    xy, dep = project(np.asarray(centre, float), B, off)
    z = zscale(dep)
    lx, ly = float(np.dot(light, r_)), float(np.dot(light, u_))
    n = np.hypot(lx, ly) or 1.0
    lx, ly = lx / n, ly / n
    rim = mix(rgb, '#000000', 0.5)
    for k, (fr, o, tint) in enumerate([(1.00, 0.00, 0.00), (0.70, 0.22, 0.28), (0.38, 0.36, 0.58)]):
        ax.add_patch(Circle((xy[0] + o * radius * lx, xy[1] + o * radius * ly), fr * radius,
                            facecolor=mix(rgb, '#ffffff', tint),
                            edgecolor=rim if k == 0 else 'none',
                            lw=0.22 if k == 0 else 0.0, zorder=z + 0.001 * (k + 1)))


def corners_of(s0, s1, off=ZERO):
    return np.array([[x, y, z] for x in (s0 - 0.5, s1 + 0.5)
                     for (y, z) in perimeter_path()[:-1]]) + np.asarray(off, float)


def zscaler(dmin, dmax):
    span = max(dmax - dmin, 1e-9)
    return lambda d: 1.0 + 998.0 * (d - dmin) / span


def draw_slice(ax, d, fi, s0, s1, ring, B, light, zscale, R_tube, off=ZERO, cage_every=1):
    draw_cage(ax, s0, s1, B, zscale, off, every=cage_every)
    draw_cylinder(ax, s0 - 0.5, s1 + 0.5, R_tube, B, light, zscale, off)
    occ = d['dens'][fi]
    n_tot = 0
    for l, (y, z) in enumerate(ring):
        for s in range(s0, s1 + 1):
            if int(occ[l, s]) <= 0:
                continue
            draw_sphere(ax, (s, y, z), 0.42, P.MOTOR, B, zscale, light, off)
            n_tot += int(occ[l, s])
    return n_tot


def draw_overview(ax, d, fi, L, s0, s1, x0, y0, W, H):
    """The whole microtubule at this instant - lane-averaged density along its full length - with
    the rendered window boxed. Returns the box's two right-hand corners, to rule across from.
    Drawn above the scene's zorder range so nothing in the render can cover it."""
    from matplotlib.patches import Rectangle
    ax.imshow(kymo(d)[fi][None, :], extent=[x0, x0 + W, y0, y0 + H], aspect='auto',
              cmap=P.cmap(), vmin=0, vmax=1, interpolation='nearest', zorder=1200)
    ax.add_patch(Rectangle((x0, y0), W, H, fill=False, ec=P.INK_DIM, lw=0.4, zorder=1201))
    bx0, bx1 = x0 + W * s0 / L, x0 + W * (s1 + 1) / L
    ax.add_patch(Rectangle((bx0, y0 - 0.45), bx1 - bx0, H + 0.9, fill=False, ec=P.INK, lw=0.8,
                           zorder=1202))
    return (bx1, y0 + H + 0.45), (bx1, y0 - 0.45)


def rule_across(ax, a, b, ta, tb):
    """The two leader lines from a boxed window to the magnified render."""
    from matplotlib.lines import Line2D
    for p0, p1 in ((a, ta), (b, tb)):
        ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], color=P.INK_DIM, lw=0.45,
                           alpha=0.85, zorder=1203))


def frame_axes(ax, pts_screen, pad=0.7):
    ax.set_xlim(pts_screen[:, 0].min() - pad, pts_screen[:, 0].max() + pad)
    ax.set_ylim(pts_screen[:, 1].min() - pad, pts_screen[:, 1].max() + pad)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_facecolor(P.GROUND)


def draw_cartoon(ax, ring, R_tube):
    """The lattice in two views, and the two moves a motor can make: a step along the tube, and a
    sidestep around it to the neighbouring lane. The axial view is the one the tube renders cannot
    show - it is what makes the 16 lanes, and therefore the sidestep, visible at all."""
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
    edge = mix(P.CAGE, '#ffffff', 0.35)

    # --- axial view: looking down the tube axis ---------------------------------------------
    for y in range(BODY_LO, BODY_HI + 1):
        for z in range(BODY_LO, BODY_HI + 1):
            ax.add_patch(Rectangle((y - 0.5, z - 0.5), 1, 1, facecolor=P.TUBE_FILL,
                                   edgecolor=edge, lw=0.4, zorder=1))
    for (y, z) in ring:
        ax.add_patch(Rectangle((y - 0.5, z - 0.5), 1, 1, facecolor='none', edgecolor=P.CAGE,
                               lw=0.45, zorder=2))
    ax.add_patch(Circle((CENTRE, CENTRE), R_tube, facecolor=P.TUBE,
                        edgecolor=mix(P.TUBE, '#ffffff', 0.25), lw=0.5, zorder=3))
    ax.text(CENTRE, CENTRE, 'MT', fontsize=6.5, ha='center', va='center', color='w', zorder=5)
    a0, a1 = (SHELL_HI, CENTRE), (SHELL_HI, CENTRE + 1)
    ax.add_patch(Circle(a0, 0.42, facecolor=P.MOTOR, edgecolor=mix(P.MOTOR, '#000000', 0.5),
                        lw=0.3, zorder=4))
    ax.add_patch(FancyArrowPatch(a0, a1, connectionstyle='arc3,rad=0.55', arrowstyle='-|>',
                                 mutation_scale=8, lw=1.0, color=P.INK, shrinkA=3, shrinkB=3,
                                 zorder=6))
    ax.text(SHELL_HI + 1.0, CENTRE + 0.6, 'sidestep', fontsize=6.5, ha='left', va='center',
            color=P.INK)
    ax.text(CENTRE, SHELL_LO - 1.35, 'axial view', fontsize=7, ha='center', color=P.INK)

    # --- side view: along the tube ------------------------------------------------------------
    X0, NC = 10.6, 6
    ax.add_patch(Rectangle((X0 - 0.5, 2.35), NC, 1.6, facecolor=P.TUBE,
                           edgecolor=mix(P.TUBE, '#ffffff', 0.25), lw=0.5, zorder=3))
    ax.text(X0 - 0.5 + 0.5 * NC, 3.15, 'MT', fontsize=6.5, ha='center', va='center', color='w',
            zorder=5)
    for i in range(NC):
        ax.add_patch(Rectangle((X0 - 0.5 + i, 3.95), 1, 1, facecolor='none', edgecolor=P.CAGE,
                               lw=0.45, zorder=2))
    b0, b1 = (X0 + 1.0, 4.45), (X0 + 2.0, 4.45)
    ax.add_patch(Circle(b0, 0.42, facecolor=P.MOTOR, edgecolor=mix(P.MOTOR, '#000000', 0.5),
                        lw=0.3, zorder=4))
    ax.add_patch(FancyArrowPatch(b0, b1, arrowstyle='-|>', mutation_scale=7, lw=0.9, color=P.INK,
                                 shrinkA=6, shrinkB=6, zorder=6))
    ax.text(X0 + 1.5, 5.35, 'step', fontsize=6.5, ha='center', color=P.INK)
    ax.annotate('', xy=(X0 - 0.5 + NC, 6.35), xytext=(X0 + 2.6, 6.35),
                arrowprops=dict(arrowstyle='-|>', lw=0.7, color=P.INK_DIM))
    ax.text(X0 + 2.4, 6.35, 'plus end', fontsize=6.5, ha='right', va='center', color=P.INK_DIM)
    ax.text(X0 - 0.5 + 0.5 * NC, SHELL_LO - 1.35, 'side view', fontsize=7, ha='center',
            color=P.INK)

    # --- scale ---------------------------------------------------------------------------------
    ax.plot([SHELL_LO - 0.5, SHELL_LO + 0.5], [0.05] * 2, '-', color=P.INK, lw=0.8)
    for xv in (SHELL_LO - 0.5, SHELL_LO + 0.5):
        ax.plot([xv, xv], [-0.1, 0.2], '-', color=P.INK, lw=0.8)
    ax.text(SHELL_LO, -0.45, r'$h$ = 10 nm', fontsize=6.5, ha='center', va='top', color=P.INK)
    ax.set_xlim(0.7, X0 - 0.5 + NC + 0.8); ax.set_ylim(-1.1, SHELL_HI + 1.1)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_facecolor(P.GROUND)


def draw_time_stack(ax, d, L, times, ring, R_tube, view=(13.0, 21.0), sites=72,
                    window_end=0, dy=45.0, ov_width=46.0, ov_height=3.0, gap=8.0,
                    time_arrow=True, verbose=True):
    """The time sequence in one axes: each slice with the whole track beside it, staggered along
    the depth axis. Returns (s0, s1). ★Every slice's label is anchored at the SAME x, the leftmost
    slice's: per-slice anchors march left with the stagger, and the last one then runs into the
    time arrow."""
    B = basis(*view)
    r_, u_, c_ = B
    light = -0.45 * r_ + 0.62 * u_ + 0.65 * c_
    light /= np.linalg.norm(light)
    s1 = (window_end or L) - 1
    s0 = max(0, s1 - sites + 1)
    n = len(times)
    offs = [np.array([0.0, (n - 1 - i) * dy, 0.0]) for i in range(n)]     # earliest farthest
    allc = np.vstack([corners_of(s0, s1, o) for o in offs])
    _, dep = project(allc, B)
    zscale = zscaler(dep.min(), dep.max())

    geom = []
    for off in offs:
        face, _ = project(np.array([[s0 - 0.5, y, z] for (y, z) in perimeter_path()[:-1]]),
                          B, off)
        top, bot = face[np.argmax(face[:, 1])], face[np.argmin(face[:, 1])]
        ov_x1 = face[:, 0].min() - gap
        geom.append((top, bot, ov_x1 - ov_width, 0.5 * (top[1] + bot[1]) - 0.5 * ov_height))
    label_x = min(g[2] for g in geom) - 1.6
    # The labels are right-aligned AT label_x and run left from it; the arrow has to clear their
    # width, which is about 16 voxels at the scale these figures are drawn.
    arrow_x = label_x - 16.5

    for i, (tt, off, g) in enumerate(zip(times, offs, geom)):
        fi = int(np.argmin(np.abs(d['times'] - tt)))
        n_tot = draw_slice(ax, d, fi, s0, s1, ring, B, light, zscale, R_tube, off, cage_every=2)
        rho = float(d['dens'][fi][:, s0:s1 + 1].mean())
        top, bot, ov_x0, ov_y0 = g
        ca, cb = draw_overview(ax, d, fi, L, s0, s1, ov_x0, ov_y0, ov_width, ov_height)
        rule_across(ax, ca, cb, top, bot)
        ax.text(label_x, ov_y0 + 0.5 * ov_height,
                '$t$ = %.0f s\n$\\rho$ = %.2f' % (d['times'][fi], rho), fontsize=7.5,
                ha='right', va='center', color=P.INK, linespacing=1.5)
        if i == n - 1:
            for xv, lab in ((ov_x0, '0'), (ov_x0 + ov_width, '10 $\\mu$m')):
                ax.text(xv, ov_y0 - 1.9, lab, fontsize=6.5, ha='center', va='top',
                        color=P.INK_DIM)
        if verbose:
            print('  t = %6.0f s  sites %d-%d  rho %.3f  %d motors'
                  % (d['times'][fi], s0, s1, rho, n_tot))

    xy_all, _ = project(allc, B)
    pts = np.vstack([xy_all, [[arrow_x - 3.2, g[3] - 3.4] for g in geom],
                     [[g[2] + ov_width, g[3] + ov_height + 1.0] for g in geom]])
    ax.set_xlim(pts[:, 0].min() - 1.0, pts[:, 0].max() + 1.0)
    ax.set_ylim(pts[:, 1].min() - 1.0, pts[:, 1].max() + 1.0)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_facecolor(P.GROUND)
    if time_arrow:
        ylo, yhi = ax.get_ylim()
        ax.annotate('', xy=(arrow_x, ylo + 0.06 * (yhi - ylo)),
                    xytext=(arrow_x, yhi - 0.06 * (yhi - ylo)),
                    arrowprops=dict(arrowstyle='-|>', lw=0.8, color=P.INK_DIM))
        ax.text(arrow_x - 2.2, 0.5 * (ylo + yhi), 'time', fontsize=7.5, rotation=90,
                va='center', ha='center', color=P.INK_DIM)
    return s0, s1


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=('time', 'space'), default='time')
    p.add_argument('--times', default='100,250,400,1200', help='time mode: the slices, in s')
    p.add_argument('--time', type=float, default=1200.0, help='space mode: the instant')
    p.add_argument('--sites', type=int, default=72, help='window width in binding sites')
    p.add_argument('--window-end', type=int, default=0, help='time mode: last site (0 = the tip)')
    p.add_argument('--dy', type=float, default=45.0, help='time mode: depth offset per slice')
    p.add_argument('--view', type=float, nargs=2, default=(13.0, 21.0), help='azimuth elevation')
    p.add_argument('--dilute-at', type=int, default=250, help='space mode: dilute window centre')
    p.add_argument('--ov-width', type=float, default=46.0, help='overview strip width')
    p.add_argument('--ov-height', type=float, default=3.0, help='overview strip height')
    p.add_argument('--gap', type=float, default=8.0, help='overview to render gap')
    p.add_argument('--out', required=True)
    A = p.parse_args()

    jcp = style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta, d = load(A.run)
    L = meta['sites']
    ring = [tuple(int(v) for v in yz) for yz in d['ring']]
    R_tube = 0.5 * (BODY_HI - BODY_LO + 1)                  # inscribed in the 3x3 body: 1.5 h
    B = basis(*A.view)
    r_, u_, c_ = B
    light = -0.45 * r_ + 0.62 * u_ + 0.65 * c_
    light /= np.linalg.norm(light)

    BAND = 0.10                      # the cartoons are their own figure now
    AXW = 0.90                       # axes width as a fraction of the figure
    # 6.5 in is the manuscript's single-column preprint text block (468 pt); jcp.FULL = 6.69 is
    # the two-column reprint width and would be scaled by 0.97 on the page. See kip3_app_fig.
    TEXTW = 6.5

    if A.mode == 'time':
        times = [float(x) for x in A.times.split(',')]
        span = None
        fig = plt.figure(figsize=(TEXTW, 3.2))
        fig.patch.set_facecolor(P.GROUND)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        draw_time_stack(ax, d, L, times, ring, R_tube, view=A.view, sites=A.sites,
                        window_end=A.window_end, dy=A.dy, ov_width=A.ov_width,
                        ov_height=A.ov_height, gap=A.gap)
    else:
        fig = plt.figure(figsize=(TEXTW, 5.2))
        fig.patch.set_facecolor(P.GROUND)
        fi = int(np.argmin(np.abs(d['times'] - A.time)))
        prof = np.convolve(kymo(d)[fi], np.ones(20) / 20, mode='same')
        above = np.flatnonzero(prof > 0.5)
        wall = int(above[0]) if len(above) else L - A.sites // 2
        half = A.sites // 2
        wins = [('dilute antenna region', A.dilute_at - half), ('the domain wall', wall - half),
                ('inside the jam', L - A.sites - 2)]
        gs = fig.add_gridspec(3, 1, left=0.015, right=0.895, top=0.975, bottom=0.045, hspace=0.06)
        for i, (name, s0) in enumerate(wins):
            ax = fig.add_subplot(gs[i, 0])
            s0 = int(np.clip(s0, 0, L - A.sites - 1)); s1 = s0 + A.sites - 1
            _, dep = project(corners_of(s0, s1), B)
            zscale = zscaler(dep.min(), dep.max())
            n_tot = draw_slice(ax, d, fi, s0, s1, ring, B, light, zscale, R_tube)
            rho = float(d['dens'][fi][:, s0:s1 + 1].mean())
            xy, _ = project(corners_of(s0, s1), B)
            frame_axes(ax, xy)
            ax.text(0.0, 0.90, '(%s)  %s' % ('abc'[i], name), transform=ax.transAxes,
                    fontsize=8.5, color=P.INK)
            ax.text(1.012, 0.74, '%.1f–%.1f $\\mu$m\n$\\rho$ = %.2f\n%d motors'
                    % (s0 * M.H_NM * 1e-3, (s1 + 1) * M.H_NM * 1e-3, rho, n_tot),
                    transform=ax.transAxes, fontsize=7, va='top', color=P.INK, family='monospace')
            print('  (%s) %-22s sites %4d-%4d  rho %.3f  %d motors'
                  % ('abc'[i], name, s0, s1, rho, n_tot))

    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None, facecolor=fig.get_facecolor())
    print('wrote %s.pdf/.png  (%s mode, view az %.0f el %.0f)'
          % (A.out, A.mode, A.view[0], A.view[1]))


if __name__ == '__main__':
    main()
