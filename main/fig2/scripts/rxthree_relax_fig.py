"""Figure: time-resolved relaxation of the TERNARY mixture with an oversize species, scored against
Brownian dynamics.

A (2 nm) and B (4 nm) at eta 0.07 each start in half the box by random sequential insertion; C (20 nm,
larger than the 10 nm voxel) sits at the centre of the empty half and is mobile. Arms, data paths and
the pooling are rxthree_data.py, shared with the supplement's correlation figure.

FOUR LETTERED BLOCKS, in the layout idiom of the application-section figures (kip3_app_fig.py,
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
fig:three-kl (SI/figS13/scripts/rxthree_kl_fig.py).

★The block letters are upright bold and the species are sans-serif math, so block D and the species
 C are different glyphs -- but block D's title does read "mobility of C", and a reader skimming for
 panel C could land on it. The letters are set well left of the panels for that reason.

The cross-species correlation moved to the supplement (rxthree_rho_fig.py): the argument here is that
the lattice reproduces the marginals and the body's mobility, and the correlation is a second,
independent statement that reads better with a frame of its own.

Drawn on the publication page (FIGURE_STYLE.md, jcp_style.PUB_RC): Arial only, 8.5 x 11 in, 0.5 in margins, each
letter and its title on one line (jcp_style.pub_header). Parameter values and equations are in the caption.
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
    # The species key goes inside, in the empty half's top corner: below the frame it costs a band of
    # height the three stacked blocks cannot spare, and here it covers nothing.
    halo = [pe.withStroke(linewidth=1.6, foreground='white')]
    for j, (key, lbl) in enumerate((('A', 'A, 2 nm'), ('B', 'B, 4 nm'), ('C', 'C, 20 nm'))):
        y = 75.5 - 6.0 * j
        ax.add_patch(Circle((45.0, y), 0.5 * SIG[key] if key != 'C' else 2.0, facecolor=COL[key],
                            edgecolor='none', zorder=6))
        ax.text(49.0, y, lbl, fontsize=jcp_style.TEXT_PT, ha='left', va='center', color=P.INK, zorder=6,
                path_effects=halo)
    ax.set_xlim(0, 80); ax.set_ylim(0, 80); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color(P.CAGE)


def marginal(ax, n, Pb, L_, pk, i, nmax, nm, handles, title=None, keyloc='upper right'):
    """One occupancy distribution: the reference in open circles, each lattice arm over it, and its own key in the
    corner the distribution leaves empty (keyloc)."""
    handles['BD'], = ax.plot(n, Pb[i], 'o-', color=REF, mfc='white', mew=1.1, ms=4.5, label='BD')
    for key, lbl, col, sty in arms:
        handles[lbl], = ax.plot(n, L_[key][pk][i], sty, color=col, mfc=col, mew=0, ms=4.0, label=lbl)
    ax.set_xlim(-0.5, nmax); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xticks(np.arange(0, nmax + 1, 10 if nmax > 20 else 4))
    ax.set_xlabel(r'occupancy $n_\mathrm{%s}$' % nm)
    ax.set_ylabel(r'$P(n_\mathrm{%s})$' % nm)
    ax.tick_params(top=True, right=True)
    if title:
        ax.set_title(title, pad=4, fontsize=jcp_style.LABEL_PT)
    top = max(np.max(Pb[i]), max(np.max(L_[k][pk][i]) for k, *_ in arms))
    ax.set_ylim(top=YHEAD * top)                      # headroom: the key sits above every curve
    add_key(ax, keyloc)


def add_key(ax, loc, extra=()):
    """The same three-entry key in every plot (plus any plot-specific entries)."""
    hs = [Line2D([], [], marker='o', ls='-', color=REF, mfc='white', mew=1.1, ms=4.5, lw=1.5, label='BD')]
    for k_, lbl, col, sty in arms:
        hs.append(Line2D([], [], marker=sty[0], ls=sty[1:], color=col, mfc=col, mew=0, ms=4.0, lw=1.5, label=lbl))
    ax.legend(handles=hs + list(extra), loc=loc, fontsize=jcp_style.TEXT_PT, frameon=False, handlelength=1.8,
              labelspacing=0.25, borderpad=0.1, handletextpad=0.4, borderaxespad=0.2)


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
    ax.plot(u[:, o1], u[:, o2], '-', color=P.INK, lw=1.0, zorder=5, path_effects=halo)
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
# Page layout, in inches from the page's top-left corner (FIGURE_STYLE.md: 8.5 x 11 in, 0.5 in margins).
# Left column: block a (the start). Right column: blocks b (equilibrium) and c (mobility). Full width: block d.
# ---------------------------------------------------------------------------------------------
J = jcp_style
M = J.MARGIN
XA, WA = M, 2.75                 # block a: header line and column
XR, WR = 3.55, J.PAGE_W - M - 3.55   # blocks b, c
YA = YB = M                      # header lines
YC = 3.78
YD = 7.66


def header(fig, letter, title, x, ytop, w):
    """pub_header, with a title wrapped onto several centred lines kept on the LETTER's baseline by its first line
    (matplotlib's 'baseline' alignment of a multi-line text is its last line)."""
    lt, tt = J.pub_header(fig, letter, title, x, ytop, w)
    n = title.count('\n')
    if n:
        tt.set_multialignment('center')
        tt.set_y(tt.get_position()[1] - n * 1.2 * J.TITLE_PT / 72.0 / J.PAGE_H)
        tt.set_x((x + 0.30 + 0.5 * (w - 0.30)) / J.PAGE_W)     # centred over the line right of the letter
    return lt, tt


def TX(fig, x, ytop, text, **kw):
    """Figure text placed in inches from the page's top-left corner."""
    return fig.text(x / J.PAGE_W, 1.0 - ytop / J.PAGE_H, text, **kw)


YHEAD = 2.0                      # headroom of the distributions: the key sits above every curve

D = RD.prepare()
grid, L, arms = D['grid'], D['L'], D['arms']
PbA, PbB, mr_b, mr_b_e = D['PbA'], D['PbB'], D['mr_b'], D['mr_b_e']
pos0, spec0, cpos0 = initial_configuration(0)
pos_grid = grid > 0            # the tau = 0 point, where the displacement is zero, is off the log panel
i_eq = len(grid) - 1

with plt.rc_context(J.PUB_RC):
    fig = J.pub_page()
    handles = {}

    # ---- a: the nonequilibrium initial condition ----------------------------------------------
    header(fig, 'a', 'Nonequilibrium start', XA, YA, WA)
    ax_cart = J.pub_ax(fig, XA + 0.12, 1.22, 2.55, 2.55)
    draw_cartoon(ax_cart, pos0, spec0, cpos0)
    ax_cart.set_title('configuration at the start', pad=5, fontsize=J.LABEL_PT)
    a0 = J.pub_ax(fig, XA + 0.62, 4.30, 2.08, 1.10)
    a1 = J.pub_ax(fig, XA + 0.62, 5.95, 2.08, 1.10)
    marginal(a0, np.arange(NMAX_A + 1), PbA, L, 'PA', 0, NMAX_A, 'A', handles,
             title='occupancy at the start')
    marginal(a1, np.arange(NMAX_B + 1), PbB, L, 'PB', 0, NMAX_B, 'B', handles)

    # ---- b: the two distributions at equilibrium ----------------------------------------------
    header(fig, 'b', 'MPD-RDDME predicts equilibrium distributions\ncloser to BD than point-particle simulations',
           XR, YB, WR)
    e0 = J.pub_ax(fig, XR + 0.55, 1.35, 1.58, 1.70)
    e1 = J.pub_ax(fig, XR + 2.75, 1.35, 1.58, 1.70)
    marginal(e0, np.arange(NMAX_A + 1), PbA, L, 'PA', i_eq, NMAX_A, 'A', handles)
    marginal(e1, np.arange(NMAX_B + 1), PbB, L, 'PB', i_eq, NMAX_B, 'B', handles)

    # ---- c: the body's mean squared displacement ----------------------------------------------
    header(fig, 'c', 'MPD-RDDME approaches the crowded BD\ndiffusivity at long times', XR, YC, WR)
    mx = J.pub_ax(fig, XR + 0.62, 4.40, WR - 0.74, 2.55)
    tt = grid[pos_grid] * H * H / D_A
    mx.plot(grid[pos_grid], 6 * D_C * tt, '-', color=P.INK_DIM, lw=1.2)
    mx.fill_between(grid[pos_grid], (mr_b - mr_b_e)[pos_grid], (mr_b + mr_b_e)[pos_grid],
                    color=REF, alpha=0.25, lw=0)
    mx.plot(grid[pos_grid], mr_b[pos_grid], 'o-', color=REF, mfc='white', mew=1.1, ms=4.5)
    for key, lbl, col, sty in arms:
        mx.plot(grid[pos_grid], L[key]['msdraw'][pos_grid], sty, color=col, mfc=col, mew=0, ms=4.0)
    mx.set_xscale('log'); mx.set_yscale('log')
    # The published window ends at tau 4 (ticks to 4, y to 600 nm^2); a longer run extends both instead of clipping.
    ymax = max(600.0, 1.5 * max(np.max(6 * D_C * tt), np.max(L['point']['msdraw'][pos_grid])))
    mx.set_xlim(grid[1] * 0.8, grid[-1] * 1.25); mx.set_ylim(0.3, ymax)
    mtk = [v for v in (0.1, 0.5, 1, 2, 4, 10, 20) if v <= grid[-1] * 1.05]
    mx.set_xticks(mtk); mx.set_xticklabels(['%g' % v for v in mtk])
    mx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    mx.tick_params(top=True, right=True, which='both')
    mx.set_xlabel(r'time $\tau$')
    mx.set_ylabel('mean squared displacement of C (nm\u00b2)')
    add_key(mx, 'lower right', [Line2D([], [], ls='-', color=P.INK_DIM, lw=1.2, label='free diffusion')])
    mx.grid(alpha=0.25, lw=0.4, which='both')

    # ---- d: the lattice configurations --------------------------------------------------------
    # The cube is the map and the slice is the measurement: see A CUBE IS CONTEXT, A SLICE IS
    # EVIDENCE in rxthree_scene3d.py. Each time gets a cube with the cut layer outlined, ruled
    # across to that layer enlarged, and the two share the colour bar at the right.
    J.pub_rule(fig, YD - 0.17)
    header(fig, 'd', 'Lattice configurations at the start and at equilibrium', M, YD, J.PAGE_W - 2 * M)
    snap = np.load(D['lat']['model'][0]['path'], allow_pickle=True)
    stau = np.asarray(snap['tau'], float)
    snap_meta = json.loads(str(snap['meta']))
    frames = [None, len(stau) - 1]      # None = the rebuilt tau = 0 start, see initial_lattice
    Bv = S3.basis(28.0, 18.0)
    light = np.array([0.35, -0.55, 0.76]); light /= np.linalg.norm(light)
    SAX = 2                                        # cut normal to the third axis, so the slab axis
    CY, CH, CW3 = 8.40, 2.00, 1.25                 # runs across the enlargement and the second up it
    for (cx, sx), fi, lab in zip(((0.50, 1.85), (3.95, 5.30)), frames, ('start', 'equilibrium')):
        eta_v, bodies = (initial_lattice(snap_meta) if fi is None else S3.cell_of(snap, fi))
        tau_v = 0.0 if fi is None else stau[fi]
        # Cut the layer the body's centre sits in, taken from the body rather than assumed: the
        # enlargement is only worth the space if the body is in it at full cross section.
        kcut = int(round(float(bodies[0][SAX]))) if len(bodies) else S3.CELL // 2
        ax3 = J.pub_ax(fig, cx, CY - 0.10, CW3, CH + 0.20)
        xy = S3.draw_scene(ax3, eta_v, bodies, Bv, light)
        if fi is not None:
            mark_path3d(ax3, snap, Bv)               # the same path in the cube, in 3D
        pl = S3.draw_slice_plane(ax3, kcut, SAX, Bv)
        ax3.set_xlim(xy[:, 0].min() - 0.4, xy[:, 0].max() + 0.4)
        ax3.set_ylim(xy[:, 1].min() - 0.4, xy[:, 1].max() + 0.4)
        ax3.set_aspect('equal'); ax3.axis('off')
        ax2 = J.pub_ax(fig, sx, CY, CH, CH)
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
                                           color=S3.HILITE, lw=1.1, zorder=0))
        TX(fig, 0.5 * (cx + sx + CH), CY - 0.12, lab, ha='center', va='bottom', fontsize=J.LABEL_PT)
        print('  scene %-11s tau %.4g: eta mean %.3f, max %.3f; cut layer %d of %d'
              % (lab, tau_v, eta_v.mean(), eta_v.max(), kcut, S3.CELL))
    sm = matplotlib.cm.ScalarMappable(cmap=P.cmap_binary(),
                                      norm=matplotlib.colors.Normalize(0, S3.ETA_MAX))
    cb = fig.colorbar(sm, cax=J.pub_ax(fig, 7.42, CY + 0.15, 0.11, CH - 0.30), ticks=[0, 0.1, 0.2, 0.3])
    cb.ax.set_yticklabels(['0', '0.1', '0.2', '0.3'])
    cb.set_label(r'bath volume fraction $\eta$', fontsize=J.TEXT_PT)
    cb.ax.tick_params(labelsize=J.TEXT_PT)
    cb.outline.set_linewidth(0.6)

    J.pub_check(fig, 'rxthree_relax')
    J.check_text_overlap(fig, 'rxthree_relax')
    for e in ('pdf', 'png'):
        os.makedirs(os.path.join(FIGD, e), exist_ok=True)
        fig.savefig(os.path.join(FIGD, e, 'rxthree_relax.' + e), dpi=300)
    J.pub_fonts(os.path.join(FIGD, 'pdf', 'rxthree_relax.pdf'), 'rxthree_relax')
    import panel_export                            # each panel alone, no letter or title: ../plots/panels/
    panel_export.export(fig, 'rxthree_relax')
print('-> rxthree_relax.{pdf,png}')
RD.report(D)
