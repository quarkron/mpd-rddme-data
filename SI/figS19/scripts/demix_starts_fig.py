"""SI figure fig:demix-starts: polymer-crowder demixing from two other starts than the main-text slab (fig:phase c).

    python demix_starts_fig.py [--install]

  a  the UNIFORM start (monomers and crowders placed at random; the block d of fig:polymer until 2026-10-09) at
     t = 0, 2, 10 s, and its chain-length panels
  b  the DROPLET start (monomers in a central sphere of half the box volume, crowders outside; demix_slab.sh
     --start droplet) at t = 0, 1, 10 s, and its chain-length panels
Each start as fig:phase c draws the slab start: the whole box in 3-D beside a two-tone slice through its middle
(demix_scene3d.panel_boxes, demix_slab_maps.panel_maps), one row per arm, one box-slice pair per time; beneath, the
chain length over time and its distribution at t = 10 s (demix_scene3d.lengths_pub). SI style (FIGURE_STYLE.md
section 5, 2026-10-19): Arial, 10/11 pt, letters only. The median / 1st-99th percentile meaning of line and band is the
caption's.
Reads SI/figS8's data and scripts. Writes ../plots/{pdf,png}/demix_starts.*; --install copies into
the manuscript's figures/.
"""
import argparse, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
F4 = os.path.join(HERE, '..', '..', 'figS8', 'scripts')            # the chain-figure modules and data (was main/fig4)
sys.path.insert(0, F4)
import fig_polymer_r2 as F                      # noqa: E402
import demix_scene3d as D3                      # noqa: E402
import demix_slab_maps as DS                    # noqa: E402
import jcp_style as J                           # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)
DROPLET = os.path.join(F4, '..', 'data', 'demix_noneq', 'droplet')

# Layout, inches from the top-left of the figure.
SW = 7.5
W, G, P = 0.95, 0.04, 0.10        # box / slice side, box-slice gap, gap between one time's pair and the next
X0 = 0.52                         # left edge of the first box (the arm names stand left of it)
Y_T = 0.42                        # block top -> top of the first row of boxes (the t = labels sit above)
LH = 1.10                         # chain-length plots: height
Y_L = 0.46                        # bottom of the snapshot rows -> top of the chain-length plots
BLOCK = Y_T + 2 * W + G + Y_L + LH + 0.50
GAP = 0.18                        # between the two blocks
SH = 2 * BLOCK + GAP
ARM_NAMES = {'finite-size': 'finite-size', 'point particles': 'point\nparticles'}   # two lines: one row is W tall


def snapshots_si(fig, root, arms, colors, y0, times):
    """Rows = arms, columns = times (each time a 3-D box beside its slice, ruled together), the colour bar at the
    right. Returns the bottom of the rows (in)."""
    XS = [X0 + c * (2 * W + G + P) for c in range(len(times))]
    rows = range(len(arms))
    boxes = [[J.pub_ax(fig, x, y0 + r * (W + G), W, W) for x in XS] for r in rows]
    planes = D3.panel_boxes(boxes, root, arms, times, colors)
    grid = [[J.pub_ax(fig, x + W + G, y0 + r * (W + G), W, W) for x in XS] for r in rows]
    H = len(arms) * (W + G) - G
    cax = J.pub_ax(fig, XS[-1] + 2 * W + G + 0.12, y0, 0.10, H)
    DS.panel_maps(grid, cax, root, arms, times, 'twotone', titles=False, labels=False, frame=D3.OUTLINE, frame_lw=1.4,
                  pub=True)
    for t in cax.texts:                # the end words clear of the end tick labels (this bar is shorter than fig:phase's)
        x, y = t.get_position()
        t.set_position((x, 1.05 if y > 0.5 else -0.05))
    for r in rows:                     # rule the plane's two right-hand corners to the slice's left-hand ones
        for c in range(len(times)):
            pl = planes[r][c]
            right = np.argsort(pl[:, 0])[-2:]
            top, bot = right[np.argsort(pl[right, 1])[::-1]]
            for src, dst in ((top, (0.0, 1.0)), (bot, (0.0, 0.0))):
                fig.add_artist(ConnectionPatch(xyA=tuple(pl[src]), coordsA=boxes[r][c].transData, xyB=dst,
                                               coordsB=grid[r][c].transAxes, color=D3.OUTLINE, lw=0.9, zorder=0))
    for x, t in zip(XS, times):
        J.pub_text(fig, x + W + G / 2, y0 - 0.06, r'$t$ = %d s' % t, ha='center', va='bottom', fontsize=J.LABEL_PT)
    for r, (_, name) in enumerate(arms):
        J.pub_text(fig, X0 - 0.06, y0 + r * (W + G) + W / 2, ARM_NAMES.get(name, name), rotation=90, ha='right',
                   va='center', ma='center', fontsize=J.LABEL_PT, linespacing=1.05)
    return y0 + H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(SW, SH)
        cols = (F.C_POLY, F.C_CROWD)
        for letter, top, root, times in (('a', 0.0, F.DEMIX, (0, 2, 10)), ('b', BLOCK + GAP, DROPLET, (0, 1, 10))):
            J.pub_letter(fig, letter, 0.0, top)
            yb = snapshots_si(fig, root, F.ARMS_D, cols, top + Y_T, times)
            D3.lengths_pub(J.pub_ax(fig, 0.95, yb + Y_L, 2.55, LH), J.pub_ax(fig, 4.60, yb + Y_L, 2.55, LH), root,
                           F.ARMS_D, cols)
        J.pub_check(fig, 'demix_starts', si=True); J.check_text_overlap(fig, 'demix_starts')
        for e in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, e), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, e, 'demix_starts.' + e), dpi=300)
        J.pub_fonts(os.path.join(PLOTS, 'pdf', 'demix_starts.pdf'), 'demix_starts')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'demix_starts')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'demix_starts.*')))
    if A.install:
        for e in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, e, 'demix_starts.' + e), os.path.join(FIGD, e, 'demix_starts.' + e))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
