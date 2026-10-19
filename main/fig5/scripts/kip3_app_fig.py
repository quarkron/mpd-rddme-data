"""The manuscript figure for the Kip3 traffic application: model, results, configurations.

    python kip3_app_fig.py --pt out/fig1c_pt_iso_1500.npz --exc out/fig1c_exc_iso_1500.npz \
        --sw out/fig1c_exc_sw_1500.npz --out out/fig_kip3_app

One figure in four lettered blocks:

    A  the model in one cartoon: the lattice in axial and side view, the tilt that drives the
       step with its staircase aligned under the binding sites it belongs to, and attachment and
       detachment as a birth-death process in one voxel.
    B  kymographs of the normalized motor density for the three arms.
    C  the tracer speed against the local motor density beneath them.
    D  the configurations themselves, as the jam front sweeps through the plus end of the track.

Only the blocks are lettered. The arms are named in the kymograph titles and in C's key, so inner
letters would add nothing the reader has to look up.

Drawn on the full 8.5 x 11 in publication page (FIGURE_STYLE.md, jcp_style.PUB_RC): Arial only,
no TeX. Rate symbols and parameter values are in the caption, not on the panels.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))
import kip3_model as M
import kip3_palette as P
from kip3_analyze import load, kymo, link_tracers_full, bin_speed
from kip3_scheme_fig import draw_scheme
from kip3_snapshot3d import draw_time_stack, BODY_LO, BODY_HI

import jcp_style as J

# Page geometry (FIGURE_STYLE.md): 8.5 x 11 in, 0.5 in margins, every position in inches from the
# page's top-left corner. Row 1 holds the model cartoon (a, a narrow left column, height-limited)
# beside the kymographs (b); the speed curve (c) and the configurations (d) each take a full row.
W = J.PAGE_W - 2 * J.MARGIN
XA, WA = J.MARGIN, 1.95                    # block a: cartoon column
XB = 2.55                                  # block b: header left edge
KX0, KX1, KGAP = 3.25, 7.05, 0.16          # kymograph frames: left of the first, right of the last
CBX, CBW = 7.17, 0.12                      # colour bar
Y1, KTOP, KH = 0.50, 1.50, 2.62            # row 1 header, kymograph top, kymograph height
YC, CTOP, CH = 4.95, 5.62, 1.45            # block c header, plot top, plot height
CX0, CW = 1.15, 6.70                       # block c plot frame
YD, DTOP = 7.73, 8.07                      # block d header, scene top (scene runs to the margin)

# The long titles are wrapped, not reworded: the first line on the letter's baseline, the rest
# beneath it, each centred over the block.
TITLE_B = ('MPD-RDDME can simulate Kip3 traffic jams in microtubule',
           'plus ends with volume exclusion and motor sidestepping')
TITLE_C = ('Motor slowdown arises emergently from MPD-RDDME',
           'simulations of Kip3 transport')
TITLE_A = 'Lattice model'
TITLE_D = 'Jam front at the plus end'
LINE = 1.2 * J.TITLE_PT / 72.0             # title line spacing, in

TITLE = {'pt': 'point\nparticles', 'exc': 'exclusion,\nno sidestepping',
         'sw': 'exclusion,\nwith sidestepping'}
KEY = {'pt': 'point particles', 'exc': 'exclusion, no sidestepping',
       'sw': 'exclusion, with sidestepping'}


def header(fig, letter, lines, x, ytop, w, center=True):
    """jcp_style.pub_header for the first line, and any further title lines centred beneath it
    (tagged as headlines too, so panel_export leaves them out of the per-panel files)."""
    if isinstance(lines, str):
        lines = (lines,)
    lt, tt = J.pub_header(fig, letter, lines[0], x, ytop, w, center=center)
    y0 = ytop + J.LETTER_PT / 72.0
    for k, ln in enumerate(lines[1:], 1):
        t = fig.text((x + 0.5 * w) / J.PAGE_W, 1.0 - (y0 + k * LINE) / J.PAGE_H, ln,
                     fontsize=J.TITLE_PT, fontweight='bold', ha='center', va='baseline', color=J.INK)
        t.set_gid('headline')
    return y0 + (len(lines) - 1) * LINE        # the last title line's baseline, in


def TX(fig, x, ytop, text, **kw):
    """Figure text placed in inches from the page's top-left corner."""
    return fig.text(x / J.PAGE_W, 1.0 - ytop / J.PAGE_H, text, **kw)


def block_b(fig, runs):
    nk = len(runs)
    kw = (KX1 - KX0 - (nk - 1) * KGAP) / nk
    for j, (k, (meta, d)) in enumerate(runs):
        ax = J.pub_ax(fig, KX0 + j * (kw + KGAP), KTOP, kw, KH)
        K = kymo(d); t = d['times']; L = K.shape[1]
        im = ax.imshow(K, aspect='auto', origin='upper', cmap=P.cmap(), vmin=0, vmax=1,
                       extent=[0, L * M.H_NM * 1e-3, t[-1], 0], interpolation='nearest')
        ax.set_title(TITLE[k], fontsize=J.LABEL_PT, linespacing=1.2, pad=5)
        ax.set_xticks([0, 5, 10] if j == nk - 1 else [0, 5])   # a '10' beside the next panel's '0' reads as one label
        ax.set_yticks([0, 500, 1000, 1500])
        if j == 0:
            ax.set_ylabel('time (s)')
        else:
            ax.set_yticklabels([])
    TX(fig, 0.5 * (KX0 + KX1), KTOP + KH + 0.27, r'position along the microtubule ($\mu$m)',
       ha='center', va='top', fontsize=J.LABEL_PT)
    cax = J.pub_ax(fig, CBX, KTOP, CBW, KH)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'normalized motor density $\rho$', fontsize=J.TEXT_PT)
    cb.outline.set_linewidth(0.6)


def block_c(fig, runs, t_min):
    from matplotlib.lines import Line2D
    ax = J.pub_ax(fig, CX0, CTOP, CW, CH)
    cols = dict(P.ARM)
    for k, (meta, d) in runs:
        v, rho, _ = link_tracers_full(d, window=40, w=10, t_min=t_min)
        Bs = bin_speed(v, rho)
        if not len(Bs):
            continue
        v0 = meta['v0_nm_s']
        ax.errorbar(Bs[:, 0], Bs[:, 1] / v0, yerr=Bs[:, 2] / v0, fmt='o', ms=5.5,
                    color=cols[k], capsize=2, elinewidth=1.0)
    ax.set_xlabel(r'local motor density $\rho$')
    ax.set_ylabel(r'$v / v_0$')
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.15)
    ax.tick_params(top=True, right=True)
    handles = [Line2D([], [], color=cols[k], marker='o', ms=5.5, lw=1.5, label=KEY[k])
               for k, _r in runs]
    # Upper right, the corner the data leave empty (above the jammed points, right of the last
    # point-particle bin). ★Not lower left: there the key spans block b's letter column, and
    # panel_export then files it as a shared piece instead of block c's.
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pt', required=True); p.add_argument('--exc', required=True)
    p.add_argument('--sw')
    p.add_argument('--snap', help='run to draw block D from; defaults to --exc')
    p.add_argument('--times', default='100,250,400,1200')
    p.add_argument('--t-min', type=float, default=1200.0)
    p.add_argument('--out', required=True)
    A = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = [(k, load(getattr(A, k))) for k in ('pt', 'exc', 'sw') if getattr(A, k)]
    meta_s, d_s = load(A.snap or A.exc)
    ring = [tuple(int(v) for v in yz) for yz in d_s['ring']]

    with plt.rc_context(J.PUB_RC):
        fig = J.pub_page()

        # ---- a: the model, b: the kymographs -------------------------------------------------
        header(fig, 'a', TITLE_A, XA, Y1, WA, center=False)   # a narrow column: title beside the letter
        ax_a = J.pub_ax(fig, XA, Y1 + 0.40, WA, KTOP + KH + 0.52 - (Y1 + 0.40))
        draw_scheme(ax_a, 'pub')
        header(fig, 'b', TITLE_B, XB, Y1, J.PAGE_W - J.MARGIN - XB)
        block_b(fig, runs)

        # ---- c: the speed ---------------------------------------------------------------------
        J.pub_rule(fig, YC - 0.12)
        header(fig, 'c', TITLE_C, J.MARGIN, YC, W)
        block_c(fig, runs, A.t_min)

        # ---- d: the configurations ------------------------------------------------------------
        J.pub_rule(fig, YD - 0.12)
        header(fig, 'd', TITLE_D, J.MARGIN, YD, W)
        ax_d = J.pub_ax(fig, J.MARGIN, DTOP, W, J.PAGE_H - J.MARGIN - DTOP)
        draw_time_stack(ax_d, d_s, meta_s['sites'], [float(x) for x in A.times.split(',')],
                        ring, 0.5 * (BODY_HI - BODY_LO + 1), fs=J.TEXT_PT, label_w=19.0)

        J.pub_check(fig, 'kip3_traffic')
        J.check_text_overlap(fig, 'kip3_traffic')
        for ext in ('pdf', 'png'):
            fig.savefig('%s.%s' % (A.out, ext), dpi=300)
        J.pub_fonts(A.out + '.pdf', 'kip3_traffic')
        import panel_export
        panel_export.export(fig, 'kip3_traffic')
    print('wrote %s.pdf/.png' % A.out)


if __name__ == '__main__':
    main()
