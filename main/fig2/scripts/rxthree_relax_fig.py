"""Figure: time-resolved relaxation of the TERNARY mixture with an oversize species, scored against
Brownian dynamics.

A (2 nm) and B (4 nm) at eta 0.07 each start in half the box by random sequential insertion; C (20 nm,
larger than the 10 nm voxel) sits at the centre of the empty half and is mobile. Arms, data paths and
the pooling are rxthree_data.py, shared with the supplement's correlation figure.

FIVE LETTERED BLOCKS, in the layout idiom of the application-section figures (kip3_app_fig.py,
tcr_app_fig.py) with one letter per claim rather than per column:

    A  the nonequilibrium initial condition in one cartoon -- the slab start drawn to scale from the
       reference's own initializer, over the subvolume grid -- with the two starting occupancy
       distributions beneath it, which is that same initial condition stated as the quantity the rest
       of the figure tracks.
    B  the two occupancy distributions at equilibrium.
    C  the body's mean squared displacement.
    D  the lattice configurations themselves, at the start and at equilibrium (rxthree_scene3d.py).

Nothing inside a block is lettered: B holds two panels, distinguished by their titles. Every plot carries its own key.
The divergence of each distribution from the Brownian reference over time (block C until 2026-10-05) is the SI figure
fig:three-kl (SI/figS12/scripts/rxthree_kl_fig.py).

★The block letters are upright bold and the species are sans-serif math, so block D and the species
 C are different glyphs -- but block D's title does read "mobility of C", and a reader skimming for
 panel C could land on it. The letters are set well left of the panels for that reason.

The cross-species correlation moved to the supplement (rxthree_rho_fig.py): the argument here is that
the lattice reproduces the marginals and the body's mobility, and the correlation is a second,
independent statement that reads better with a frame of its own.

Runs in the lm2.6 container; TeX Live must be on PATH or the type will not match the other figures.
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rxthree_data as RD
import jcp_style
import kip3_palette as P
import rxthree_scene3d as S3
from rxthree_data import (FIGD, NMAX_A, NMAX_B, H, D_A, D_C, SIG, TEXTW, REF, ARMS, COL,
                          initial_configuration, initial_lattice)

FIGH = 7.30


# The offsets that put a note beside its letter: 0.24 in to the right of the letter's left edge,
# and 0.035 in down, which is the difference in cap height between 12 pt and 8 pt and so lands the
# two on one baseline rather than on one top edge.
NOTE_LINE = 0.185          # in: the letter's top to the top of the sentence on the line below it


def LETTER(fig, blk, c, x, ytop, note=None):
    """A block letter, given in inches from the top-left like AX, and the block's one-line
    conclusion on the line BENEATH it, so nothing shares a horizontal level with a letter. Placed 0.28 in above the top of the first panel of its block, which is
    clear of that panel's title.

    The note shares the letter's row because there is nowhere else for it: five caption lines under
    five blocks would add three quarters of an inch to a figure already 7.3 in deep, whose own
    caption runs twelve lines on the same page."""
    fig.text(x / TEXTW, 1.0 - ytop / FIGH, blk(c.lower()), fontsize=12, fontweight='bold',
             ha='left', va='top')
    if note:
        jcp_style.block_note(fig, x / TEXTW, 1.0 - (ytop + NOTE_LINE) / FIGH, note)


MIN_GAP = 0.22          # in: the least white space a stacked pair may leave, measured


def check_gaps(fig, columns):
    """Measure the white space between stacked blocks and report it, every run.

    The clearances this layout is built on -- 0.34 in for a row of tick labels plus an axis label,
    0.17 for a title -- are guesses until something measures them, and the failure they guard against
    is silent: a label that just touches the block below still renders, it only looks joined.
    get_tightbbox carries every decoration an axes owns, so the distance from one block's lowest
    extent to the next block's highest is the white space actually on the page. Printed always,
    and called out when it falls under MIN_GAP."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    worst = None
    for col, items in columns:
        spans = []
        for name, axs in items:
            bb = [a.get_tightbbox(r) for a in axs]
            top = min(b.y0 for b in bb); bot = max(b.y1 for b in bb)      # display y is up
            spans.append((name, (fig.get_figheight() * dpi - bot) / dpi,
                          (fig.get_figheight() * dpi - top) / dpi))       # -> inches from the top
        for (n0, _t0, b0), (n1, t1, _b1) in zip(spans, spans[1:]):
            gap = t1 - b0
            flag = '  << under %.2f' % MIN_GAP if gap < MIN_GAP else ''
            print('  gap %-8s %s -> %-8s %.3f in%s' % (col, n0, n1, gap, flag))
            if worst is None or gap < worst[0]:
                worst = (gap, '%s %s->%s' % (col, n0, n1))
    if worst:
        print('  tightest block gap: %.3f in (%s)' % worst)


def AX(x, ytop, w, h):
    """An axes rectangle given in INCHES from the top-left of the canvas.

    The layout was fractional and every time a block changed height the gaps had to be re-derived,
    because a fraction is only a length once the canvas height is fixed. What actually constrains
    this figure is absolute: a panel needs about 0.34 in beneath it for its tick labels and x label
    and 0.17 in above it for its title, whatever the canvas is. Stating the layout in inches makes
    those clearances readable, and the canvas height FIGH is then where the last block ends plus a
    margin.

    THE BLOCK PITCH. Between two stacked blocks the frames must clear 0.34 + 0.17 for the label and
    title that sit between them, plus white space of their own, or the lower block reads as a
    continuation of the upper one. At 0.25 in of white that is a frame-to-frame pitch of 0.76, and
    the right-hand column uses 1.75 = 0.95 of panel + 0.80 of gap. The earlier 0.52 pitch left only
    0.01 in between B's x label and C's title, which is why they looked joined."""
    return [x / TEXTW, 1.0 - (ytop + h) / FIGH, w / TEXTW, h / FIGH]


# ---------------------------------------------------------------------------------------------
# Block A: the cartoon
# ---------------------------------------------------------------------------------------------
def draw_cartoon(ax, pos0, spec0, cpos0, slab=4.0):
    """The starting configuration in a 8 nm slab through the body, drawn as the particles themselves at
    their true diameters in the box's own nanometre coordinates -- the reference's configuration to
    scale, not a schematic of it -- over the subvolume grid the lattice arms actually carry.

    Every artist carries an explicit zorder. With the defaults the shaded half is a patch and the
    particles a collection, and matplotlib 3.4 draws the patch last, which hid the particles
    completely (2026-09-08)."""
    ax.axvspan(0, 40, color='#eef2f7', lw=0, zorder=0)
    # The subvolume grid, faint and under everything: this is a lattice method, and the cartoon is
    # otherwise indistinguishable from an off-lattice configuration -- which is exactly what the
    # reference it was generated from is. One line every h = 10 nm, so the reader can count the eight
    # subvolumes a side that block C then draws as cubes.
    for t in np.arange(0.0, 80.0 + 1e-9, 10.0):
        ax.axvline(t, color=P.CAGE, lw=0.3, alpha=0.55, zorder=1)
        ax.axhline(t, color=P.CAGE, lw=0.3, alpha=0.55, zorder=1)
    ax.axvline(40, color=P.CAGE, lw=0.7, ls=(0, (3, 2)), zorder=1.5)
    for sp, key in ((0, 'A'), (1, 'B')):
        sel = (spec0 == sp) & (np.abs(pos0[:, 2] - 40.0) < slab)
        d = SIG[key]
        ax.add_collection(EllipseCollection(widths=d, heights=d, angles=0, units='xy',
                                            offsets=pos0[sel, :2], transOffset=ax.transData,
                                            facecolors=COL[key], edgecolors='none', linewidths=0,
                                            alpha=0.85, zorder=2 + sp, rasterized=True))
        print('  cartoon %s: %d particles of %g nm in the slab' % (key, int(sel.sum()), d))
    # The body, shaded the same way it is in block C so the two read as the same object.
    cx, cy = cpos0[0, 0], cpos0[0, 1]
    for k, (fr, o, tint) in enumerate([(1.00, 0.00, 0.00), (0.70, 0.22, 0.26), (0.38, 0.36, 0.55)]):
        ax.add_patch(Circle((cx + o * 0.5 * SIG['C'] * 0.30, cy + o * 0.5 * SIG['C'] * 0.65),
                            fr * 0.5 * SIG['C'], facecolor=S3.mix(COL['C'], '#ffffff', tint),
                            edgecolor=S3.mix(COL['C'], '#000000', 0.45) if k == 0 else 'none',
                            lw=0.4 if k == 0 else 0.0, zorder=5 + 0.01 * k))
    # The slab's density is called out under the frame rather than inside it: at this scale a label
    # laid over nine hundred particles needs a halo to be read at all, and the halo then punches a
    # hole in the configuration it is describing.
    halo = [pe.withStroke(linewidth=1.6, foreground='white')]
    ax.text(20, -5, r'$\eta = 0.14$', fontsize=7.0, ha='center', va='top', color=P.INK,
            zorder=6, clip_on=False)
    ax.text(40, -12, r'$D_\mathsf{A},\,D_\mathsf{B},\,D_\mathsf{C} = 0.1,\ 0.05,\ 0.01\ \mu$m$^2$/s', fontsize=6.4,
            ha='center', va='top', color=P.INK, zorder=6, clip_on=False)   # bare D, Stokes-Einstein with size
    # The species key goes inside, in the empty half's top corner: below the frame it costs a band of
    # height the three stacked blocks cannot spare, and here it covers nothing.
    for j, (key, lbl) in enumerate((('A', r'$\mathsf{A}$, 2 nm'), ('B', r'$\mathsf{B}$, 4 nm'),
                                    ('C', r'$\mathsf{C}$, 20 nm'))):
        y = 76.5 - 5.5 * j
        ax.add_patch(Circle((44.0, y), 0.5 * SIG[key] if key != 'C' else 1.6, facecolor=COL[key],
                            edgecolor='none', zorder=6))
        ax.text(47.5, y, lbl, fontsize=6.4, ha='left', va='center', color=P.INK, zorder=6,
                path_effects=halo)
    ax.set_xlim(0, 80); ax.set_ylim(0, 80); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(P.CAGE)


def marginal(ax, n, Pb, L_, pk, i, nmax, nm, handles, title=None, keyloc='upper right'):
    """One occupancy distribution: the reference in open circles, each lattice arm over it, and its own key in the
    corner the distribution leaves empty (keyloc)."""
    handles['BD'], = ax.plot(n, Pb[i], 'o-', color=REF, mfc='white', mew=0.9, ms=3.0, label='BD')
    for key, lbl, col, sty in arms:
        handles[lbl], = ax.plot(n, L_[key][pk][i], sty, color=col, mfc=col, mew=0, ms=2.8, label=lbl)
    ax.set_xlim(-0.5, nmax); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xticks(np.arange(0, nmax + 1, 10 if nmax > 20 else 4))
    ax.set_xlabel(r'occupancy $n_%s$' % nm)
    ax.set_ylabel(r'$P(n_%s)$' % nm)
    if title:
        ax.set_title(title, pad=3, fontsize=8.0)
    top = max(np.max(Pb[i]), max(np.max(L_[k][pk][i]) for k, *_ in arms))
    ax.set_ylim(top=2.0 * top)                      # headroom: the key sits above every curve
    add_key(ax, keyloc)


def add_key(ax, loc, extra=()):
    """The same three-entry key in every plot (plus any plot-specific entries)."""
    hs = [Line2D([], [], marker='o', ls='-', color=REF, mfc='white', mew=0.9, ms=3.0, lw=1.0, label='BD')]
    for k_, lbl, col, sty in arms:
        hs.append(Line2D([], [], marker=sty[0], ls=sty[1:], color=col, mfc=col, mew=0, ms=2.8, lw=1.0, label=lbl))
    ax.legend(handles=hs + list(extra), loc=loc, fontsize=6.3, frameon=False, handlelength=1.3,
              labelspacing=0.2, borderpad=0.1, handletextpad=0.35, borderaxespad=0.15)


def mark_path(ax, snap, axis, body=0):
    """On the equilibrium enlargement: the drawn body's start (dashed outline, true diameter), its path projected on the
    slice (unwrapped across the periodic box), and an arrow from start to end, so the reader sees that C moved."""
    from matplotlib.patches import Circle as _C
    p = np.asarray(snap['posC'], float)[:, body]
    n = int(json.loads(str(snap['meta']))['N'])
    dp = np.diff(p, axis=0); dp -= n * np.round(dp / n)
    u = np.concatenate([p[:1], p[:1] + np.cumsum(dp, axis=0)]) + 0.5        # cell coordinates, voxel centres
    o1, o2 = [a for a in range(3) if a != axis]
    R = 0.5 * SIG['C'] / H
    ax.add_patch(_C((u[0, o1], u[0, o2]), R, fill=False, ls=(0, (2.5, 1.5)), lw=0.9, ec=P.INK, zorder=4))
    halo = [pe.withStroke(linewidth=2.2, foreground='white')]              # readable over the orange disc
    ax.plot(u[:, o1], u[:, o2], '-', color=P.INK, lw=0.7, zorder=5, path_effects=halo)
    ax.plot([u[0, o1]], [u[0, o2]], 'o', color=P.INK, ms=2.5, zorder=6, path_effects=halo)
    an = ax.annotate('', xy=(u[-1, o1], u[-1, o2]), xytext=(u[0, o1], u[0, o2]), zorder=6,
                     arrowprops=dict(arrowstyle='-|>', color=P.INK, lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=9))
    an.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.0, foreground='white')])
    # (no in-panel key: the caption says what the dashed outline and the line are)
    print('  C path (body %d): displacement %.2f voxels = %.1f nm' % (body, np.linalg.norm(u[-1] - u[0]), H * np.linalg.norm(u[-1] - u[0])))


def mark_path3d(ax, snap, B, body=0):
    """On the equilibrium cube: the drawn body's recorded 3D path (unwrapped) projected by the cube's own camera, its
    start as a dashed outline at true size (orthographic, so the outline is a circle of the body's radius)."""
    from matplotlib.patches import Circle as _C
    p = np.asarray(snap['posC'], float)[:, body]
    n = int(json.loads(str(snap['meta']))['N'])
    dp = np.diff(p, axis=0); dp -= n * np.round(dp / n)
    u = np.concatenate([p[:1], p[:1] + np.cumsum(dp, axis=0)]) + 0.5
    xy, _ = S3.project(u, B)
    halo = [pe.withStroke(linewidth=2.4, foreground='white')]
    ax.add_patch(_C(tuple(xy[0]), 0.5 * SIG['C'] / H, fill=False, ls=(0, (2.0, 1.2)), lw=0.9, ec=P.INK, zorder=20))
    ax.plot(xy[:, 0], xy[:, 1], '-', color=P.INK, lw=0.85, zorder=21, path_effects=halo)
    an = ax.annotate('', xy=tuple(xy[-1]), xytext=tuple(xy[0]), zorder=22,
                     arrowprops=dict(arrowstyle='-|>', color=P.INK, lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=9))
    an.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.0, foreground='white')])


# ---------------------------------------------------------------------------------------------
D = RD.prepare()
grid, L, arms = D['grid'], D['L'], D['arms']
PbA, PbB, mr_b, mr_b_e = D['PbA'], D['PbB'], D['mr_b'], D['mr_b_e']
pos0, spec0, cpos0 = initial_configuration(0)
rc = RD.style(sans=True)
xt = [0, 0.5, 1, 2, 4]; xtl = ['0', '0.5', '1', '2', '4']
pos_grid = grid > 0            # the tau = 0 point, where the displacement is zero, is off the log panel
i_eq = len(grid) - 1

with plt.rc_context(rc):
    # ★Under usetex matplotlib ignores fontweight: the block letters have to carry the weight in the
    # TeX, and stay plain when usetex is off. Same device as kip3_app_fig.py.
    blk = ((lambda c: r'\textbf{%s}' % c) if matplotlib.rcParams['text.usetex'] else (lambda c: c))
    fig = plt.figure(figsize=(TEXTW, FIGH))
    handles = {}

    # ---- A: the nonequilibrium initial condition ----------------------------------------------
    ax_cart = fig.add_axes(AX(0.30, 0.40, 1.72, 1.62))
    draw_cartoon(ax_cart, pos0, spec0, cpos0)
    ax_cart.set_title('nonequilibrium initial condition', pad=3, fontsize=7.6)
    # 2.46, not 2.32: the cartoon's eta label hangs 0.10 in below its frame, and at 2.32 it and
    # this panel's title sat on the same line and read as one collided caption.
    a0 = fig.add_axes(AX(0.52, 2.63, 1.45, 0.80))
    a1 = fig.add_axes(AX(0.52, 4.06, 1.45, 0.80))
    marginal(a0, np.arange(NMAX_A + 1), PbA, L, 'PA', 0, NMAX_A, r'\mathsf{A}', handles,
             title=r'start, $\tau = 0$')
    marginal(a1, np.arange(NMAX_B + 1), PbB, L, 'PB', 0, NMAX_B, r'\mathsf{B}', handles)
    LETTER(fig, blk, 'A', 0.08, 0.02)

    # ---- B, C, D: the measurement --------------------------------------------------------------
    e0 = fig.add_axes(AX(2.83, 0.68, 1.40, 0.95)); e1 = fig.add_axes(AX(5.04, 0.68, 1.40, 0.95))
    marginal(e0, np.arange(NMAX_A + 1), PbA, L, 'PA', i_eq, NMAX_A, r'\mathsf{A}', handles,
             title=r'equilibrium, $\tau = %.3g$' % grid[i_eq])
    marginal(e1, np.arange(NMAX_B + 1), PbB, L, 'PB', i_eq, NMAX_B, r'\mathsf{B}', handles,
             title=r'equilibrium, $\tau = %.3g$' % grid[i_eq])
    LETTER(fig, blk, 'B', 2.32, 0.02,
           note='MPD-RDDME predicts equilibrium distributions closer to BD\nthan point-particle simulations')
    mx = fig.add_axes(AX(2.83, 2.49, 3.61, 2.37))          # the right column below B, down to A's last panel
    tt = grid[pos_grid] * H * H / D_A
    mx.plot(grid[pos_grid], 6 * D_C * tt, '-', color=P.INK_DIM, lw=0.8)
    mx.fill_between(grid[pos_grid], (mr_b - mr_b_e)[pos_grid], (mr_b + mr_b_e)[pos_grid],
                    color=REF, alpha=0.25, lw=0)
    mx.plot(grid[pos_grid], mr_b[pos_grid], 'o-', color=REF, mfc='white', mew=0.9, ms=3.0)
    for key, lbl, col, sty in arms:
        mx.plot(grid[pos_grid], L[key]['msdraw'][pos_grid], sty, color=col, mfc=col, mew=0, ms=3.0)
    mx.set_xscale('log'); mx.set_yscale('log')
    # The published window ends at tau 4 (ticks to 4, y to 600 nm^2); a longer run extends both instead of clipping.
    ymax = max(600.0, 1.5 * max(np.max(6 * D_C * tt), np.max(L['point']['msdraw'][pos_grid])))
    mx.set_xlim(grid[1] * 0.8, grid[-1] * 1.25); mx.set_ylim(0.3, ymax)
    mtk = [v for v in (0.1, 0.5, 1, 2, 4, 10, 20) if v <= grid[-1] * 1.05]
    mx.set_xticks(mtk); mx.set_xticklabels(['%g' % v for v in mtk])
    mx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    mx.set_xlabel(r'time $\tau = D_\mathsf{A} t/h^{2}$')
    mx.set_ylabel(r'$\langle\Delta r_\mathsf{C}^{2}\rangle$ (nm$^{2}$)')
    add_key(mx, 'lower right', [Line2D([], [], ls='-', color=P.INK_DIM, lw=0.8, label=r'free, $6D_\mathsf{C}t$')])
    mx.set_title(r'mobility of $\mathsf{C}$', pad=3, fontsize=8.0)
    mx.grid(alpha=0.25, lw=0.4, which='both')


    LETTER(fig, blk, 'C', 2.32, 2.02,
           note='MPD-RDDME approaches the crowded BD diffusivity at long times')

    # ---- E: the lattice configurations --------------------------------------------------------
    # The cube is the map and the slice is the measurement: see A CUBE IS CONTEXT, A SLICE IS
    # EVIDENCE in rxthree_scene3d.py. Each time gets a small cube with the cut layer outlined, ruled
    # across to that layer enlarged, and the two share the colour bar at the right.
    snap = np.load(D['lat']['model'][0]['path'], allow_pickle=True)
    stau = np.asarray(snap['tau'], float)
    snap_meta = json.loads(str(snap['meta']))
    frames = [None, len(stau) - 1]      # None = the rebuilt tau = 0 start, see initial_lattice
    Bv = S3.basis(28.0, 18.0)
    light = np.array([0.35, -0.55, 0.76]); light /= np.linalg.norm(light)
    e_axes = []
    SAX = 2                                        # cut normal to the third axis, so the slab axis
    CY, CH = 5.65, 1.50                            # runs across the enlargement and the second up it
    for (cx, sx), fi, lab in zip(((0.20, 1.38), (3.25, 4.43)), frames, ('start', 'equilibrium')):
        eta_v, bodies = (initial_lattice(snap_meta) if fi is None else S3.cell_of(snap, fi))
        tau_v = 0.0 if fi is None else stau[fi]
        # Cut the layer the body's centre sits in, taken from the body rather than assumed: the
        # enlargement is only worth the space if the body is in it at full cross section.
        kcut = int(round(float(bodies[0][SAX]))) if len(bodies) else S3.CELL // 2
        ax3 = fig.add_axes(AX(cx, CY, 1.00, CH)); e_axes.append(ax3)
        xy = S3.draw_scene(ax3, eta_v, bodies, Bv, light)
        if fi is not None:
            mark_path3d(ax3, snap, Bv)               # the same path in the cube, in 3D
        pl = S3.draw_slice_plane(ax3, kcut, SAX, Bv)
        ax3.set_xlim(xy[:, 0].min() - 0.4, xy[:, 0].max() + 0.4)
        ax3.set_ylim(xy[:, 1].min() - 0.4, xy[:, 1].max() + 0.4)
        ax3.set_aspect('equal'); ax3.axis('off')
        ax2 = fig.add_axes(AX(sx, CY, CH, CH)); e_axes.append(ax2)
        S3.draw_slice2d(ax2, eta_v, bodies, kcut, SAX, P.cmap_binary())
        if fi is not None:
            mark_path(ax2, snap, SAX)                # C moved: where it started, the path it took, where it is now
        # Rule the pair together from the plane's two right-hand corners to the enlargement's two
        # left-hand ones, which is the magnifier convention: it says "this expands to that", it does
        # not claim corner-for-corner identity.
        rightmost = np.argsort(pl[:, 0])[-2:]
        top, bot = rightmost[np.argsort(pl[rightmost, 1])[::-1]]
        for src, dst in ((top, (0.0, float(S3.CELL))), (bot, (0.0, 0.0))):
            fig.add_artist(ConnectionPatch(xyA=tuple(pl[src]), coordsA=ax3.transData,
                                           xyB=dst, coordsB=ax2.transData,
                                           color=S3.HILITE, lw=0.9, zorder=0))
        fig.text((cx + 0.50 * (sx - cx + CH)) / TEXTW, 1.0 - (CY - 0.17) / FIGH,
                 r'%s, $\tau = %.3g$' % (lab, tau_v), ha='center', va='bottom', fontsize=8.0)
        print('  scene %-11s tau %.4g: eta mean %.3f, max %.3f; cut layer %d of %d'
              % (lab, tau_v, eta_v.mean(), eta_v.max(), kcut, S3.CELL))
    sm = matplotlib.cm.ScalarMappable(cmap=P.cmap_binary(),
                                      norm=matplotlib.colors.Normalize(0, S3.ETA_MAX))
    cb = fig.colorbar(sm, cax=fig.add_axes(AX(6.05, CY + 0.20, 0.10, CH - 0.40)))
    cb.set_label(r'bath volume fraction $\eta$', fontsize=8.0)
    cb.ax.tick_params(labelsize=7.5)
    LETTER(fig, blk, 'D', 0.08, CY - 0.20)

    # Both columns are checked down into E, which spans the width: a full-width block can be
    # crowded by either column above it, and only the left one was ever the tighter.
    check_gaps(fig, [('left ', [('A.cartoon', [ax_cart]), ('A.P(nA)', [a0]), ('A.P(nB)', [a1]),
                                ('D', e_axes)]),
                     ('right', [('B', [e0, e1]), ('C', [mx]), ('D', e_axes)])])
    jcp_style.check_width(fig, TEXTW, 'rxthree_relax')
    jcp_style.check_text_overlap(fig, 'rxthree_relax')
    for e in ('pdf', 'png'):
        os.makedirs(os.path.join(FIGD, e), exist_ok=True)
        fig.savefig(os.path.join(FIGD, e, 'rxthree_relax.' + e), dpi=300, bbox_inches=None)
    import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import panel_export
    panel_export.export(fig, 'rxthree_relax')
print('-> rxthree_relax.{pdf,png}')
RD.report(D)
