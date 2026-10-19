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

Publication style (FIGURE_STYLE.md section 5, 2026-10-19): Arial through jcp_style.PUB_RC (no usetex), bold 15 pt
letters and no headlines, nothing under 10 pt, no parameter values in the panels. The drive strengths that used to
head B and C (f0 = 1 kT/um and f0 = 0) and the force tags on the potential (f0, 2 f0) are the caption's now.
"""
import argparse, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common')))   # vex-rddme/common

import jcp_style                                                            # noqa: E402
import kip3_palette as P                                                     # noqa: E402
import synapse_maps_fig as MAP_                                             # noqa: E402
import tcr_scheme_fig as SCH                                                # noqa: E402

COARSEN = 2          # 80 nm display cells, within 10% of the reference mesh polygon

# Page geometry in inches from the top-left (FIGURE_STYLE.md section 5: at most 7.5 in wide).
W = 7.38
MAP = 1.06           # side of one square map panel
GAP = 0.07           # between map panels
MX = 0.92            # left edge of the map grids (row label + y label + ticks to its left)
LDY = 0.30           # a letter's top above the first thing in its block


def maps_block(fig, dp, mp, de, me, times, ytop, vmax, cmap, row_labels, show_times):
    """A 2 x N block of map panels whose top edge is ytop (in), left edge MX."""
    nt = len(times)
    im = None
    for ri, (dens, meta, lab) in enumerate(((dp, mp, row_labels[0]), (de, me, row_labels[1]))):
        y = ytop + ri * (MAP + GAP)
        for ti in range(nt):
            a = jcp_style.pub_ax(fig, MX + ti * (MAP + GAP), y, MAP, MAP)
            im = MAP_.draw_panel(a, dens[ti], meta, COARSEN, vmax, cmap,
                                 title=(('%g s' % round(float(times[ti]), 2)) if (ri == 0 and show_times) else None),
                                 xticklabels=(ri == 1), yticklabels=(ti == 0))
            if ti == 0:
                a.set_ylabel('$y$ (µm)', labelpad=2)
            if ri == 1 and ti == nt // 2:
                a.set_xlabel('$x$ (µm)')
        # Row names outside the tick labels, centred on their row.
        jcp_style.pub_text(fig, 0.14, y + 0.5 * MAP, lab, rotation=90, ha='center', va='center')
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--point', required=True)
    p.add_argument('--excluded', required=True)
    p.add_argument('--f0-point', required=True)
    p.add_argument('--f0-excluded', required=True)
    p.add_argument('--times', default='0,2,10,100,200')
    p.add_argument('--vmax', type=float, default=0.0)
    p.add_argument('--out', required=True)
    A = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update(jcp_style.PUB_RC)       # Arial, no usetex (not SCH.style: that is the LaTeX path)
    cmap = MAP_.house_cmap()

    def prep(pp, pe):
        dp, mp = MAP_.load(pp)
        de, me = MAP_.load(pe)
        if not np.allclose(dp['map_times'], de['map_times']):
            raise SystemExit('%s and %s stored maps at different times' % (pp, pe))
        want = [float(x) for x in A.times.split(',')]
        keep = np.array([int(np.argmin(np.abs(dp['map_times'] - w))) for w in want])
        if len(set(keep.tolist())) != len(keep):
            raise SystemExit('--times picked the same stored frame twice')
        return (MAP_.to_density(MAP_.coarsen(dp['maps'][keep], COARSEN), mp, COARSEN), mp,
                MAP_.to_density(MAP_.coarsen(de['maps'][keep], COARSEN), me, COARSEN), me,
                dp['map_times'][keep])

    b_dp, b_mp, b_de, b_me, times = prep(A.point, A.excluded)
    c_dp, c_mp, c_de, c_me, times_c = prep(A.f0_point, A.f0_excluded)
    if not np.allclose(times, times_c):
        raise SystemExit('the driven and undriven runs stored maps at different times')
    vmax = A.vmax or float(np.nanpercentile(b_dp[-1], 99.5))

    # Vertical plan, in inches from the top: block a (the slab on the left, the edge-on contact and the potential
    # stacked on the right), then b and c, each two rows of maps; the colour bar spans b and c.
    A_TOP = LDY
    B_TOP = 3.30                     # top of b's first map row (its time titles sit above it)
    BLOCK = 2 * MAP + GAP
    C_TOP = B_TOP + BLOCK + 0.82     # b's x tick labels and x label, then c's letter row
    H = C_TOP + BLOCK + 0.52
    fig = jcp_style.si_page(W, H)

    # ---- a: the model --------------------------------------------------------------------
    SCH.draw_top_iso(fig.add_axes([0.0, 1.0 - (A_TOP + 2.25) / H, 3.45 / W, 2.25 / H]))
    SX, SY, SW, SH = 3.55, A_TOP, 3.55, 1.05
    ax_side = jcp_style.pub_ax(fig, SX, SY, SW, SH)
    y_span = (SCH.Y_APC + SCH.NM_MEMB + 1.0) - (SCH.Y_TC - SCH.NM_MEMB - 5.0)
    SCH.draw_side(ax_side, y_span * SW / SH)
    ax_side.set_xticks([]); ax_side.set_yticks([])
    for sp in ax_side.spines.values():
        sp.set_visible(False)
    ax_phi = jcp_style.pub_ax(fig, 4.45, A_TOP + 1.38, 2.75, 0.80)
    SCH.draw_potential(ax_phi)
    jcp_style.pub_letter(fig, 'a', 0.0, 0.0)

    # ---- b, c: driven and undriven -------------------------------------------------------
    rows = ('point particles', 'volume excluded')
    maps_block(fig, b_dp, b_mp, b_de, b_me, times, B_TOP, vmax, cmap, rows, True)
    jcp_style.pub_letter(fig, 'b', 0.0, B_TOP - 0.22 - LDY)
    im = maps_block(fig, c_dp, c_mp, c_de, c_me, times, C_TOP, vmax, cmap, rows, False)
    jcp_style.pub_letter(fig, 'c', 0.0, C_TOP - LDY)
    # b and c differ only in the drive: say which in words beside the letter (the values are the caption's)
    for ytop, lab in ((B_TOP - 0.22 - LDY, 'with the actin-flow potential'), (C_TOP - LDY, 'without the potential')):
        jcp_style.pub_text(fig, 0.30, ytop + jcp_style.LETTER_PT / 72.0, lab, ha='left', va='baseline',
                           fontsize=jcp_style.LABEL_PT)

    nt = len(times)
    cx = MX + nt * MAP + (nt - 1) * GAP + 0.12
    cax = jcp_style.pub_ax(fig, cx, B_TOP, 0.13, C_TOP + BLOCK - B_TOP)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('TCR-pMHC surface density (per µm²)')   # plain Arial: a mathtext exponent sits loose here

    jcp_style.pub_check(fig, 'tcrpmhc', si=True)
    jcp_style.check_text_overlap(fig, 'tcrpmhc')

    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), dpi=300)
    jcp_style.pub_fonts(A.out + '.pdf', 'tcrpmhc')
    import panel_export                            # each panel alone, no letter: ../plots/panels/
    panel_export.export(fig, 'tcrpmhc')
    print('wrote %s.pdf and %s.png  (shared colour axis 0 to %.0f um^-2)' % (A.out, A.out, vmax))
    for lab, dp, de in (('B, driven', b_dp, b_de), ('C, undriven', c_dp, c_de)):
        print('  %-12s peak point %7.0f   peak excluded %7.0f   ratio %.2f'
              % (lab, np.nanmax(dp[-1]), np.nanmax(de[-1]),
                 np.nanmax(dp[-1]) / max(np.nanmax(de[-1]), 1e-9)))


if __name__ == '__main__':
    main()
