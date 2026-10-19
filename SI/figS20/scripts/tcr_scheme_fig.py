"""The synapse model in two cartoons: the contact in side view with the binding reaction, and the
contact zone from above with the drive that acts on the product.

    python tcr_scheme_fig.py --out out/fig_scheme

LAID OUT AFTER DHARAN & FARAGO, BUT IT IS NOT THEIR MODEL
    Their Fig. 1 has the same two views, and the geometry below is theirs: the contact radius, the
    actin-depletion core at 2 um, the pSMAC edge at 4 um, and a centripetal force of magnitude f0
    inside that edge and 2 f0 beyond it. Four things differ, and the drawing is arranged so that a
    reader can see which:

    1. THE BINDING IS EXPLICIT. Their type A points are pre-formed bonds; there is no chemistry in
       their model at all. Here TCR and pMHC are separate diffusing species and the complex is a
       product, formed and destroyed by a reversible bimolecular reaction. Panel (a) is mostly
       about that, and it is the part of the figure with no counterpart in theirs.
    2. ONLY THE PRODUCT IS DRIVEN. Free TCR and pMHC feel no active force, so they carry no arrow
       in panel (b). That is Isaacson & Zhang's species choice and it is kept here.
    3. NO LFA1-ICAM1 AND NO PINNING POINTS. Their figure carries both. The long complex is absent
       from this model, and the immobile actin pinning obstacles were measured here to change
       nothing at their coverage, so drawing them would imply a mechanism this model does not use.
    4. THE MEMBRANES ARE FLAT. Theirs undulate because membrane elasticity is their driving
       mechanism. This model has no membrane elasticity: the contact is one subvolume thick and
       the gap is fixed, so a wavy membrane here would draw a physics that is not in the solver.

COLOUR
    From the shared ramp in the Kip3 microtubule study, so the two applications in the paper look
    like one paper. Reactants take the two ends of the series palette and the product takes the
    deep blue that the density panels use for the volume-excluded arm.
"""
import argparse, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common')))   # vex-rddme/common
import kip3_palette as P                                            # noqa: E402
import synapse_model as M                                           # noqa: E402

TCR, PMHC, CPLX = P.ARM['pt'], P.ARM['sw'], P.ARM['exc']
GAP_NM = 15.0                      # the complex span, and the reaction radius
FS = 10                            # every label of the cartoons (FIGURE_STYLE.md: nothing under 10 pt)


def mix(rgb, other, t):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(rgb)), np.array(to_rgb(other))
    return tuple(np.clip((1 - t) * a + t * b, 0, 1))


def style(sans=False):
    """House geometry and type. jcp_style.apply() leaves usetex off, which silently renders every
    label in DejaVu mathtext while the rest of the manuscript is Computer Modern, so probe for
    latex and switch it on as the other figure scripts here do. sans=True selects the sans-serif
    variant (opt-in per figure, see jcp_style.sans)."""
    import matplotlib, subprocess
    matplotlib.use('Agg')
    import jcp_style
    jcp_style.apply()
    tex = True
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=10)
        matplotlib.rcParams.update({'text.usetex': True,
                                    'font.serif': ['Computer Modern Roman']})
        print('typeset with LaTeX Computer Modern')
    except Exception:
        tex = False
        matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
        print('MATHTEXT FALLBACK - fonts will not match the manuscript')
    if sans:
        matplotlib.rcParams.update(jcp_style.sans(
            {'text.latex.preamble': matplotlib.rcParams.get('text.latex.preamble', '')}, tex))
    return jcp_style


# ---------------------------------------------------------------- panel (a): the contact, edge on
# Drawn to scale in nanometres, with the axes locked to equal aspect, so the domain cartoons keep
# their proportions and the 15 nm gap is the real 15 nm. The x range is set from the axes box
# ratio in main(); change one and the other must follow or the shapes shear.
NM_GAP = 15.0            # membrane-to-membrane, the complex span and the reaction radius
NM_MEMB = 4.0            # bilayer thickness
Y_TC, Y_APC = 0.0, NM_GAP


def bilayer(ax, y0, x0, x1, flip=False):
    """A lipid bilayer: two rows of head groups with the tail region between them."""
    from matplotlib.patches import Rectangle
    head = mix(P.INK, '#ffffff', 0.42)
    tail = mix(P.INK, '#ffffff', 0.80)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, NM_MEMB, facecolor=tail, edgecolor='none', zorder=2))
    xs = np.arange(x0 + 0.55, x1, 1.30)
    for yy in (y0 + 0.42, y0 + NM_MEMB - 0.42):
        ax.plot(xs, np.full_like(xs, yy), 'o', ms=1.9, color=head,
                markeredgecolor='none', zorder=3)


def _key_pts(cx, y0, w, h, r, up=True):
    """A block with a semicircular tab on its far edge, and its complement.

    up=True   the T-cell side: body from y0 upward, tab bulging further up.
    up=False  the APC side: body from y0 downward, and the same semicircle bulging UP into the
              body, i.e. a bite taken out of its lower edge.

    The two are generated from one expression on purpose. The tab and the socket are the same
    arc, so a bound pair nests exactly when the receptor's far edge meets the ligand's, with no
    fudge factor to keep in step if the sizes change.
    """
    s = 1.0 if up else -1.0
    left, right, far = cx - 0.5 * w, cx + 0.5 * w, y0 + s * h
    th = np.linspace(np.pi, 0.0, 48)
    arc = [(cx + r * np.cos(t), far + r * np.sin(t)) for t in th]
    return [(left, y0), (left, far), (cx - r, far)] + arc + [(right, far), (right, y0)]


def _piece(ax, pts, color, z=5, lw=0.5):
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                         edgecolor=mix(color, '#000000', 0.42), lw=lw,
                         joinstyle='round', zorder=z))


TCR_W, TCR_H, TAB_R = 6.2, 7.0, 1.9
MHC_W, MHC_H = 7.4, NM_GAP - 7.0          # so a bound pair spans the cleft exactly


def tcr(ax, x, z=5, color=None):
    """The receptor: a block carrying the key."""
    c = color or TCR
    ax.plot([x, x], [Y_TC - NM_MEMB * 0.55, Y_TC + 0.8], '-', color=c, lw=1.8,
            solid_capstyle='round', zorder=z - 1)
    _piece(ax, _key_pts(x, Y_TC, TCR_W, TCR_H, TAB_R, up=True), c, z)


def pmhc(ax, x, z=5, color=None):
    """The ligand: the same block with the groove the key fits."""
    c = color or PMHC
    ax.plot([x, x], [Y_APC + NM_MEMB * 0.55, Y_APC - 0.8], '-', color=c, lw=1.8,
            solid_capstyle='round', zorder=z - 1)
    _piece(ax, _key_pts(x, Y_APC, MHC_W, MHC_H, TAB_R, up=False), c, z)


def draw_side(ax, x1, complexes=3):
    """The contact, edge on. Content is placed at FRACTIONS of x1 rather than at fixed
    nanometre positions: the axes is equal-aspect, so x1 is set by the axes-box ratio, and a
    fixed layout either overflows a short wide box or leaves half of a tall one empty."""
    from matplotlib.patches import FancyArrowPatch
    f = lambda u: u * x1
    bilayer(ax, Y_APC, f(0.004), f(0.925))
    bilayer(ax, Y_TC - NM_MEMB, f(0.004), f(0.925))
    for y, lab in ((Y_APC + NM_MEMB * 0.5, 'APC'), (Y_TC - NM_MEMB * 0.5, 'T cell')):
        ax.text(f(0.935), y, lab, fontsize=FS, ha='left', va='center', color=P.INK)

    # unbound, and too far apart to react
    tcr(ax, f(0.075))
    pmhc(ax, f(0.175))
    ax.text(f(0.075), Y_TC + TCR_H + TAB_R + 0.9, 'TCR', fontsize=FS,
            ha='center', va='bottom', color=TCR)
    ax.text(f(0.175), Y_APC - MHC_H - 0.9, 'pMHC', fontsize=FS,
            ha='center', va='top', color=PMHC)

    # the reaction
    xa, xb = f(0.275), f(0.455)
    ax.add_patch(FancyArrowPatch((xa, 8.6), (xb, 8.6), arrowstyle='-|>', mutation_scale=7,
                                 lw=0.9, color=P.INK, shrinkA=0, shrinkB=0, zorder=6))
    ax.add_patch(FancyArrowPatch((xb, 6.4), (xa, 6.4), arrowstyle='-|>', mutation_scale=7,
                                 lw=0.9, color=P.INK, shrinkA=0, shrinkB=0, zorder=6))
    ax.text(0.5 * (xa + xb), 9.3, r'$k_{\mathrm{on}}$', fontsize=FS, ha='center',
            va='bottom', color=P.INK)
    ax.text(0.5 * (xa + xb), 5.7, r'$k_{\mathrm{off}}$', fontsize=FS, ha='center',
            va='top', color=P.INK)

    # bound: the docked pair, on a pale capsule that says the lattice carries it as ONE species
    # The product is drawn in the PRODUCT colour, the one the density panels and the drive
    # markers use, so the reader meets the same blue in the cartoon and in the results. The two
    # halves differ only in tint, enough to keep the interlock legible. The pale capsule that
    # used to say "one species" is gone: the colour says it.
    xs = [f(u) for u in np.linspace(0.575, 0.865, complexes)]
    for x in xs:
        tcr(ax, x, z=6, color=CPLX)
        pmhc(ax, x, z=6, color=mix(CPLX, '#ffffff', 0.30))
    ax.text(0.5 * (xs[0] + xs[-1]), Y_TC - NM_MEMB - 1.2,
            'TCR-pMHC, one lattice species',
            fontsize=FS, ha='center', va='top', color=CPLX)

    ax.set_xlim(0, x1)
    ax.set_ylim(Y_TC - NM_MEMB - 5.0, Y_APC + NM_MEMB + 1.0)
    ax.set_aspect('equal')


# ---------------------------------------------------------------- panel (b): the zone, from above
def draw_top(ax, annotate=True, n_free=26):
    from matplotlib.patches import Circle, FancyArrowPatch
    R, RC, RP = M.R_DISK_UM, M.R_CORE_UM, M.R_PSMAC_UM
    rng = np.random.default_rng(4)

    # a representative lattice. The production grid is 288 x 288 at 40 nm and cannot be drawn;
    # this is 14 cells across so that the discretization is visible as a fact about the model.
    g = np.linspace(-R, R, 15)
    for v in g:
        ax.plot([v, v], [-R, R], '-', color=mix(P.CAGE, '#ffffff', 0.55), lw=0.35, zorder=1)
        ax.plot([-R, R], [v, v], '-', color=mix(P.CAGE, '#ffffff', 0.55), lw=0.35, zorder=1)
    ax.add_patch(Circle((0, 0), R, facecolor='none', edgecolor=P.INK, lw=0.9, zorder=4))
    for rad, ls in ((RC, ':'), (RP, '--')):
        ax.add_patch(Circle((0, 0), rad, facecolor='none', edgecolor=P.INK_DIM,
                            lw=0.8, ls=ls, zorder=4))

    # free reactants, scattered and undriven
    for color, n in ((TCR, n_free), (PMHC, n_free)):
        k = 0
        while k < n:
            x, y = rng.uniform(-R, R, 2)
            if RC * 1.05 < np.hypot(x, y) < R * 0.95:
                ax.plot([x], [y], 'o', ms=1.9, color=color, zorder=5); k += 1

    # the product, driven inward: short arrows inside the pSMAC edge, long ones beyond it
    # The two rings are offset by half a step. Sharing the eight angles put every inner molecule
    # on the ray its outer neighbour's arrow sweeps down, so the long arrows ran through them.
    base = np.linspace(0, 2 * np.pi, 9)[:-1]
    half = 0.5 * (base[1] - base[0])
    for r0, scale, angs in ((0.62 * (RC + RP), 1.0, base + half),
                            (0.5 * (RP + R) + 0.35, 2.0, base)):
        for ang in angs:
            x, y = r0 * np.cos(ang), r0 * np.sin(ang)
            # Length carries the force: one unit inside the pSMAC edge, two beyond it. At the
            # previous 0.42 um the ratio was there but neither arrow read as an arrow.
            L = 0.95 * scale
            ax.plot([x], [y], 'o', ms=3.2, color=CPLX, zorder=6)
            ax.add_patch(FancyArrowPatch((x, y), (x - L * np.cos(ang), y - L * np.sin(ang)),
                                         arrowstyle='-|>', mutation_scale=6 + 2 * scale,
                                         lw=0.7 + 0.35 * scale, color=P.INK, zorder=6,
                                         shrinkA=2.0, shrinkB=0))
    if annotate:
        ax.text(0, 0, r'$\mathsf{TCR}\textrm{-}\mathsf{pMHC}$' + '\n' + 'accumulates',
                fontsize=6.5, ha='center', va='center', color=P.INK_DIM, linespacing=1.3)

    # A corner key rather than labels on the circles: the drive arrows ring the annulus at every
    # 45 degrees, so any in-panel label on a circle lands on one of them.
    if annotate:
        kx, ky, dy = -R * 1.04, R * 1.02, 0.42
        for i, (ls, lab) in enumerate(((':', r'$R_{\mathrm{C}}=2\ \mu$m'),
                                       ('--', r'$R_{\mathrm{P}}=4\ \mu$m'))):
            yy = ky - i * dy
            ax.plot([kx, kx + 0.62], [yy, yy], ls, color=P.INK_DIM, lw=0.8, zorder=6)
            ax.text(kx + 0.78, yy, lab, fontsize=6.5, ha='left', va='center', color=P.INK_DIM)
    ax.set_xlim(-R * 1.10, R * 1.10); ax.set_ylim(-R * 1.10, R * 1.28)
    ax.set_aspect('equal')


# ---------------------------------------------------------------- the zone in an isometric view (2026-10-09)
def _iso(x, y, z=0.0, az=-35.0, el=28.0):
    """Orthographic projection of the zone plane (x, y; z up, um) at azimuth az and elevation el."""
    a, e = np.radians(az), np.radians(el)
    X = x * np.cos(a) - y * np.sin(a)
    Y = (x * np.sin(a) + y * np.cos(a)) * np.sin(e) + z * np.cos(e)
    return X, Y


def draw_top_iso(ax, n_free=30, slab=0.55):
    """The contact zone as a 3-D isometric slab: the lattice on its top face (a representative 14 x 14 grid, as in
    draw_top), the actin-depletion (dotted) and pSMAC (dashed) rings, free TCR and pMHC, and the complex driven inward
    (arrow length = local force, one unit inside the pSMAC edge, two beyond it)."""
    from matplotlib.patches import Polygon, FancyArrowPatch
    R, RC, RP = M.R_DISK_UM, M.R_CORE_UM, M.R_PSMAC_UM
    rng = np.random.default_rng(4)
    th = np.linspace(0, 2 * np.pi, 361)
    cx, cy = R * np.cos(th), R * np.sin(th)
    # the slab: side wall (front half), then the top face
    X0, Y0 = _iso(cx, cy, 0.0); X1, Y1 = _iso(cx, cy, -slab)
    front = Y1 < np.median(Y1)
    k = np.argsort(X0[front])
    side = np.r_[np.c_[X0[front][k], Y0[front][k]], np.c_[X1[front][k], Y1[front][k]][::-1]]
    ax.add_patch(Polygon(side, closed=True, fc=mix(P.CAGE, '#ffffff', 0.55), ec=P.INK, lw=0.6, zorder=1))
    ax.add_patch(Polygon(np.c_[X0, Y0], closed=True, fc='#f6f8fb', ec=P.INK, lw=0.9, zorder=2))
    # lattice lines, clipped to the disk
    for v in np.linspace(-R, R, 15):
        h = np.sqrt(max(R * R - v * v, 0.0))
        for xs, ys in (((v, v), (-h, h)), ((-h, h), (v, v))):
            X, Y = _iso(np.array(xs), np.array(ys))
            ax.plot(X, Y, '-', color=mix(P.CAGE, '#ffffff', 0.35), lw=0.35, zorder=3)
    for rad, ls in ((RC, ':'), (RP, '--')):
        X, Y = _iso(rad * np.cos(th), rad * np.sin(th))
        ax.plot(X, Y, ls, color=P.INK_DIM, lw=0.9, zorder=4)
    for color, n in ((TCR, n_free), (PMHC, n_free)):
        m = 0
        while m < n:
            x, y = rng.uniform(-R, R, 2)
            if RC * 1.05 < np.hypot(x, y) < R * 0.95:
                X, Y = _iso(x, y); ax.plot([X], [Y], 'o', ms=2.4, color=color, zorder=5); m += 1
    base = np.linspace(0, 2 * np.pi, 9)[:-1]
    half = 0.5 * (base[1] - base[0])
    for r0, scale, angs in ((0.62 * (RC + RP), 1.0, base + half), (0.5 * (RP + R) + 0.35, 2.0, base)):
        for ang in angs:
            x, y = r0 * np.cos(ang), r0 * np.sin(ang)
            L = 0.95 * scale
            Xa, Ya = _iso(x, y); Xb, Yb = _iso(x - L * np.cos(ang), y - L * np.sin(ang))
            ax.plot([Xa], [Ya], 'o', ms=4.0, color=CPLX, zorder=6)
            ax.add_patch(FancyArrowPatch((Xa, Ya), (Xb, Yb), arrowstyle='-|>', mutation_scale=6 + 2 * scale,
                                         lw=0.7 + 0.35 * scale, color=P.INK, zorder=6, shrinkA=2.0, shrinkB=0))
    Xe, Ye = _iso(cx, cy, -slab)
    ax.set_xlim(X0.min() - 0.3, X0.max() + 0.3); ax.set_ylim(min(Ye.min(), Y0.min()) - 0.3, Y0.max() + 0.3)
    ax.set_aspect('equal'); ax.axis('off')


# ---------------------------------------------------------------- the drive, under panel (b)
def draw_potential(ax):
    R, RP = M.R_DISK_UM, M.R_PSMAC_UM
    r = np.linspace(0, R, 400)
    phi = np.where(r <= RP, r, RP + 2.0 * (r - RP))
    ax.plot(r, phi, '-', color=CPLX, lw=1.5, zorder=3)
    ax.plot(-r, phi, '-', color=CPLX, lw=1.5, zorder=3)
    for s in (1, -1):
        ax.axvline(s * RP, color=P.INK_DIM, lw=0.8, ls='--', zorder=2)
        ax.axvline(s * M.R_CORE_UM, color=P.INK_DIM, lw=0.8, ls=':', zorder=2)
    # One complex per region of the ramp, with the arrow it feels; the two lengths are in the
    # ratio of the two slopes. They sit in the clear space inside the V rather than on the curve,
    # which leaves the arrows room to be read at this size. What ties a molecule to its region is
    # the pair of guide lines it sits between, not contact with the branch.
    from matplotlib.patches import FancyArrowPatch
    # The force magnitudes (f0 inside the pSMAC edge, 2 f0 beyond it) are the caption's: the arrows carry the
    # ratio by their length, and the panel names no parameter (FIGURE_STYLE.md section 4).
    Y_MOL = 1.35
    for r0, scale in ((3.0, 1.0), (5.1, 2.0)):
        ax.plot([r0], [Y_MOL], 'o', ms=5.0, color=CPLX, zorder=6)
        # Short enough that each arrow stays inside the region whose slope it reports; longer,
        # the outer one reached across the pSMAC edge and read as belonging to the inner one.
        L = 0.50 * scale
        ax.add_patch(FancyArrowPatch((r0, Y_MOL), (r0 - L, Y_MOL), arrowstyle='-|>',
                                     mutation_scale=7 + 2 * scale, lw=0.8 + 0.4 * scale,
                                     color=P.INK, zorder=6, shrinkA=2.5, shrinkB=0))
    ax.set_xlim(-R * 1.08, R * 1.08); ax.set_ylim(0, 9.2)
    ax.set_xlabel('radius $r$ (\u00b5m)')
    ax.set_ylabel(r'potential ($k_{\mathrm{B}}T$)')
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([0, 4, 8])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default='out/fig_scheme')
    p.add_argument('--width', type=float, default=6.5)
    p.add_argument('--labels', type=int, default=0)
    A = p.parse_args()

    style()
    import matplotlib.pyplot as plt
    H = 4.8
    SIDE = [0.045, 0.690, 0.935, 0.292]
    fig = plt.figure(figsize=(A.width, H))
    ax_side = fig.add_axes(SIDE)
    ax_top = fig.add_axes([0.055, 0.050, 0.425, 0.600])
    ax_phi = fig.add_axes([0.625, 0.140, 0.350, 0.410])

    # Equal aspect means the data range must carry the axes box ratio itself, or matplotlib
    # letterboxes the drawing inside the box it was given.
    y_span = (Y_APC + NM_MEMB + 1.0) - (Y_TC - NM_MEMB - 5.0)
    draw_side(ax_side, y_span * (SIDE[2] * A.width) / (SIDE[3] * H))
    draw_top(ax_top)
    draw_potential(ax_phi)
    for a in (ax_side, ax_top):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    if A.labels:
        for a, lab, xy in ((ax_side, 'a', (0.0, 1.0)), (ax_top, 'b', (0.0, 1.0)),
                           (ax_phi, 'c', (-0.30, 1.06))):
            a.text(xy[0], xy[1], '(%s)' % lab, transform=a.transAxes, fontsize=8,
                   ha='left', va='top', color=P.INK)

    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None)
    print('wrote %s.pdf and %s.png' % (A.out, A.out))


if __name__ == '__main__':
    main()
