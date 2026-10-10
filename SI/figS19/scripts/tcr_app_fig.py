"""The manuscript figure for the TCR-pMHC application: model, driven result, undriven control.

    python tcr_app_fig.py --point out/M_react_point.npz --excluded out/M_react_excluded.npz \
        --f0-point out/F0_react_point.npz --f0-excluded out/F0_react_excluded.npz \
        --out out/fig_tcr_app

One figure in three lettered blocks, built the way kip3_app_fig.py is: the drawing code lives in
the component scripts and is imported here, so the standalone figures and this one cannot drift.

    A  the model: the contact edge on with the binding reaction, the contact zone from above with
       the drive, and the potential the product feels, on the same radial axis as the zone.
    B  under the centripetal ramp: complex profiles at five times, point particles above and
       volume exclusion below.
    C  the same model with the ramp switched off. Nothing accumulates and the two descriptions
       become indistinguishable, which is what says the separation in B is the ramp acting
       through exclusion rather than the exclusion machinery on its own.

★B AND C SHARE ONE COLOUR AXIS, taken from the driven point-particle arm. Giving C its own range
would stretch it to fill the bar and invite a reader to compare a panel in B against a panel in C
as though the scales matched. The cost is that C sits low on the bar, which is the honest picture.

Only the blocks are lettered: rows are named at the left of B and C and the times are the column
titles, so inner letters would add nothing to look up.
"""
import argparse, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common')))   # vex-rddme/common

import jcp_style                                                            # noqa: E402
import kip3_palette as P                                                     # noqa: E402
import synapse_maps_fig as MAP                                              # noqa: E402
import tcr_scheme_fig as SCH                                                # noqa: E402

TEXTW = 6.5          # the 468 pt text width of the manuscript's single-column preprint layout
COARSEN = 2          # 80 nm display cells, within 10% of the reference mesh polygon

# Block A's side view reaches the top edge, so its one-line conclusion has nowhere to sit. The
# canvas grows by NOTE_BAND at the TOP and every rectangle keeps its distance from the BOTTOM, so
# nothing on the page moves or changes size: a strip appears above A and nothing else shifts.
# B and C keep their own letter rows, where the note has to stop short of the centred drive banner.
NOTE_BAND = 0.0

# The sentence goes on the line BENEATH the letter, 0.185 in down: nothing shares a horizontal
# level with a block letter.
NOTE_LINE = 0.185


def block(fig, x, y, letter, bold, note=None, figh=None):
    fig.text(x, y, bold(letter.lower()), fontsize=13, fontweight='bold', ha='left', va='top')
    if note:
        jcp_style.block_note(fig, x, y - NOTE_LINE / figh, note)


def maps_block(fig, dp, mp, de, me, times, rect, vmax, cmap, row_labels, show_times,
               banner=None, banner_dy=0.026):
    """A 2 x N block of map panels in the figure rectangle [left, bottom, right, top]."""
    left, bottom, right, top = rect
    nt = len(times)
    gs = fig.add_gridspec(2, nt, left=left, right=right, top=top, bottom=bottom,
                          wspace=0.06, hspace=0.06)
    im = None
    for ri, (dens, meta, lab) in enumerate(((dp, mp, row_labels[0]), (de, me, row_labels[1]))):
        for ti in range(nt):
            a = fig.add_subplot(gs[ri, ti])
            im = MAP.draw_panel(a, dens[ti], meta, COARSEN, vmax, cmap,
                                title=(('$t = %g$ s' % round(float(times[ti]), 2))
                                       if (ri == 0 and show_times) else None),
                                xticklabels=(ri == 1), yticklabels=(ti == 0))
            if ti == 0:
                a.set_ylabel(r'$y$ ($\mu$m)')
            if ri == 1 and ti == nt // 2:
                a.set_xlabel(r'$x$ ($\mu$m)')
        # Row captions sit outside the tick labels, placed from the real axes rectangle.
        bb = fig.axes[-1].get_position()
        fig.text(left - 0.088, 0.5 * (bottom + top) + (0.25 if ri == 0 else -0.25) * (top - bottom),
                 lab, rotation=90, ha='center', va='center', fontsize=7.5)
    if banner:
        # Centred over the grid, above the column titles when there are any. The clearance from
        # whatever sits above is checked in main(), not assumed: the band between the potential's
        # x label and block B's titles is only about a fifth of an inch.
        fig.text(0.5 * (left + right), top + banner_dy, banner,
                 ha='center', va='bottom', fontsize=8)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--point', required=True)
    p.add_argument('--excluded', required=True)
    p.add_argument('--f0-point', required=True)
    p.add_argument('--f0-excluded', required=True)
    p.add_argument('--times', default='0,2,10,100,200')
    p.add_argument('--vmax', type=float, default=0.0)
    p.add_argument('--height', type=float, default=8.30,
                   help='figure height in inches; the caption must fit under it on '
                        'the same page, since a float cannot break')
    p.add_argument('--out', required=True)
    A = p.parse_args()

    SCH.style(sans=True)
    import matplotlib
    import matplotlib.pyplot as plt

    # ★Under usetex matplotlib ignores fontweight, so the block letters carry their own \textbf.
    bold = ((lambda c: r'\textbf{%s}' % c) if matplotlib.rcParams['text.usetex']
            else (lambda c: c))
    cmap = MAP.house_cmap()

    def prep(pp, pe):
        dp, mp = MAP.load(pp)
        de, me = MAP.load(pe)
        if not np.allclose(dp['map_times'], de['map_times']):
            raise SystemExit('%s and %s stored maps at different times' % (pp, pe))
        want = [float(x) for x in A.times.split(',')]
        keep = np.array([int(np.argmin(np.abs(dp['map_times'] - w))) for w in want])
        if len(set(keep.tolist())) != len(keep):
            raise SystemExit('--times picked the same stored frame twice')
        return (MAP.to_density(MAP.coarsen(dp['maps'][keep], COARSEN), mp, COARSEN), mp,
                MAP.to_density(MAP.coarsen(de['maps'][keep], COARSEN), me, COARSEN), me,
                dp['map_times'][keep])

    b_dp, b_mp, b_de, b_me, times = prep(A.point, A.excluded)
    c_dp, c_mp, c_de, c_me, times_c = prep(A.f0_point, A.f0_excluded)
    if not np.allclose(times, times_c):
        raise SystemExit('the driven and undriven runs stored maps at different times')
    vmax = A.vmax or float(np.nanpercentile(b_dp[-1], 99.5))

    # ★Height is set by what the PAGE leaves, not by what looks comfortable. The text block is
    # 665.5 pt; a float cannot break across pages, so figure + caption must fit inside it. At the
    # 8.2 in this was first drawn at, only ~3 caption lines fit and the caption ran off the page,
    # the same failure Figure 2 had. 7.2 in leaves about 6 lines.
    # H_DRAW is the height the layout below is written in; the canvas is that plus the note band,
    # and R()/YB() map the drawing's fractions onto it.
    H_DRAW = A.height
    H = H_DRAW + NOTE_BAND
    YS = H_DRAW / H

    def R(rect):
        x, y, w, h = rect
        return [x, y * YS, w, h * YS]

    def YB(y):
        return y * YS

    fig = plt.figure(figsize=(TEXTW, H))

    # 2026-10-09 layout, in INCHES from the top (IN): block A = the zone as a 3-D isometric slab on the left (the largest
    # element), the edge-on contact and the potential stacked on the right at reduced width; more white between blocks.
    def IN(x, ytop, w, h):
        return [x / TEXTW, 1.0 - (ytop + h) / H, w / TEXTW, h / H]

    def YT(ytop):
        return 1.0 - ytop / H

    # ---- A: the model --------------------------------------------------------------------
    SCH.draw_top_iso(fig.add_axes(IN(0.05, 0.22, 3.10, 2.05)))
    SIDE = IN(3.30, 0.24, 3.15, 0.80)
    ax_side = fig.add_axes(SIDE)
    y_span = (SCH.Y_APC + SCH.NM_MEMB + 1.0) - (SCH.Y_TC - SCH.NM_MEMB - 5.0)
    SCH.draw_side(ax_side, y_span * (SIDE[2] * TEXTW) / (SIDE[3] * H))
    ax_phi = fig.add_axes(IN(3.85, 1.30, 2.45, 0.80))
    SCH.draw_potential(ax_phi)
    ax_side.set_xticks([]); ax_side.set_yticks([])
    for sp in ax_side.spines.values():
        sp.set_visible(False)
    block(fig, 0.006, YT(0.02), 'A', bold, figh=H)

    # ---- B, C: driven and undriven -------------------------------------------------------
    rows = ('point particles', 'volume excluded')
    MH = 1.83                                                    # two rows of square map panels
    B_TOP, C_TOP = 3.25, 6.10
    maps_block(fig, b_dp, b_mp, b_de, b_me, times, [0.140, YT(B_TOP + MH), 0.845, YT(B_TOP)], vmax, cmap, rows, True,
               banner=r'$f_0 = 1\ k_{\mathrm{B}}T\ \mu\mathrm{m}^{-1}$', banner_dy=0.17 / H)
    block(fig, 0.006, YT(2.62), 'B', bold, figh=H,
          note='MPD-RDDME simulates the volume-excluded aggregation of TCR-pMHC in the actin depletion zone')
    im = maps_block(fig, c_dp, c_mp, c_de, c_me, times, [0.140, YT(C_TOP + MH), 0.845, YT(C_TOP)], vmax, cmap, rows,
                    False, banner=r'$f_0 = 0$', banner_dy=0.06 / H)
    block(fig, 0.006, YT(5.55), 'C', bold, figh=H,
          note='Without a potential representing actin flow, TCR-pMHC diffuse uniformly in the contact zone')

    cax = fig.add_axes([0.862, YT(C_TOP + MH), 0.016, (C_TOP + MH - B_TOP) / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'TCR-pMHC surface density ($\mu$m$^{-2}$)')

    jcp_style.check_width(fig, TEXTW, 'tcrpmhc')
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    arts = list(fig.texts)
    for a in fig.axes:
        arts += [a.xaxis.label, a.yaxis.label, a.title]
    boxes = [(t.get_text(), t.get_window_extent(renderer=rend))
             for t in arts if t.get_text().strip()]
    for m in range(len(boxes)):
        for n in range(m + 1, len(boxes)):
            (ta, ba), (tb, bb) = boxes[m], boxes[n]
            if ba.x0 < bb.x1 and bb.x0 < ba.x1 and ba.y0 < bb.y1 and bb.y0 < ba.y1:
                print('WARNING: figure texts overlap: %r and %r' % (ta, tb))

    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None)
    import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import panel_export
    panel_export.export(fig, 'tcrpmhc')
    print('wrote %s.pdf and %s.png  (shared colour axis 0 to %.0f um^-2)' % (A.out, A.out, vmax))
    for lab, dp, de in (('B, driven', b_dp, b_de), ('C, undriven', c_dp, c_de)):
        print('  %-12s peak point %7.0f   peak excluded %7.0f   ratio %.2f'
              % (lab, np.nanmax(dp[-1]), np.nanmax(de[-1]),
                 np.nanmax(dp[-1]) / max(np.nanmax(de[-1]), 1e-9)))


if __name__ == '__main__':
    main()
