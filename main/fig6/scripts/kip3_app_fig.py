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

Runs in the lm2.6 container; TeX Live must be on PATH or the type will not match the other
figures (kip3_analyze.style warns if it is missing).
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M
import kip3_palette as P
from kip3_analyze import load, kymo, link_tracers_full, bin_speed, style, SHORT, _key
from kip3_scheme_fig import draw_scheme
from kip3_snapshot3d import draw_time_stack, BODY_LO, BODY_HI

# ★6.5 in is the 468 pt text width of the manuscript's single-column preprint layout, NOT
# jcp_style.FULL (6.69 in, the two-column reprint width). Drawn at 6.69 the figure is scaled by
# 0.97 on the page and every 8 pt label falls under the AIP floor - the same note
# fig_ribosome_arms.py carries.
TEXTW = 6.5

# Blocks A and B both reach the top edge -- A's cartoon starts there and B's kymograph titles are
# two lines tall -- so their one-line conclusions have nowhere to sit. The canvas grows by
# NOTE_BAND at the TOP and every rectangle keeps its distance from the BOTTOM through R()/YB(),
# so nothing on the page moves or changes size: a strip simply appears above A and B, which is the
# letter row C and D already have.
FIGH_DRAW = 6.3
NOTE_BAND = 0.58
FIGH = FIGH_DRAW + NOTE_BAND
YS = FIGH_DRAW / FIGH


def R(rect):
    """An axes rectangle written in the drawing's own height, mapped onto the taller canvas."""
    x, y, w, h = rect
    return [x, y * YS, w, h * YS]


def YB(y):
    """A figure-fraction y (matplotlib measures from the bottom), mapped the same way."""
    return y * YS


TITLE = {'pt': 'point\nparticles', 'exc': 'exclusion,\nno sidestepping',
         'sw': 'exclusion,\nwith sidestepping'}
# The key inside the speed panel uses short forms: the kymograph titles above it already carry
# the full names, and at full length the key runs into the points it sits beside.
KEY = {'pt': 'point', 'exc': 'no sidestepping', 'sw': 'with sidestepping'}


# The sentence goes on the line BENEATH the letter: nothing shares a horizontal level with a
# block letter, which is also what stops block a's bullet reading as the start of block b's line.
NOTE_LINE = 0.185 / FIGH


def LETTER(fig, blk, c, x, y, note=None):
    """A lowercase block letter, alone on its line, and the block's conclusion beneath it."""
    import jcp_style
    fig.text(x, y, blk(c.lower()), fontsize=13, fontweight='bold', ha='left', va='top')
    if note:
        jcp_style.block_note(fig, x, y - NOTE_LINE, note)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pt', required=True); p.add_argument('--exc', required=True)
    p.add_argument('--sw')
    p.add_argument('--snap', help='run to draw block D from; defaults to --exc')
    p.add_argument('--times', default='100,250,400,1200')
    p.add_argument('--t-min', type=float, default=1200.0)
    p.add_argument('--out', required=True)
    A = p.parse_args()

    jcp = style(sans=True)
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ★Under usetex matplotlib ignores fontweight: the weight has to be in the TeX, so the block
    # letters are wrapped in \textbf here and left plain when usetex is off.
    blk = ((lambda c: r'\textbf{%s}' % c) if matplotlib.rcParams['text.usetex']
           else (lambda c: c))

    runs = [(k, load(getattr(A, k))) for k in ('pt', 'exc', 'sw') if getattr(A, k)]
    cols = dict(P.ARM)

    # 6.6 in leaves room for the caption on one page; block D has ~0.4 in of slack at
    # this height, since its scene is width-limited.
    fig = plt.figure(figsize=(TEXTW, FIGH))

    # ---- A: the model ------------------------------------------------------------------------
    # The cartoon axes must end left of where the kymographs' y LABEL falls, not merely left of
    # their frame.
    ax_a = fig.add_axes(R([0.0, 0.445, 0.25, 0.545]))
    draw_scheme(ax_a, 'tall')
    # Only B and C carry a headline: A is the model and D the configurations. The band above stays,
    # since B's own row is taken by kymograph titles two lines tall.
    LETTER(fig, blk, 'A', 0.012, 0.99)

    # ---- B: the kymographs, C: the speed -----------------------------------------------------
    gsk = fig.add_gridspec(1, len(runs), left=0.355, right=0.875, top=YB(0.955),
                           bottom=YB(0.695), wspace=0.15)
    for j, (k, (meta, d)) in enumerate(runs):
        ax = fig.add_subplot(gsk[0, j])
        K = kymo(d); t = d['times']; L = K.shape[1]
        im = ax.imshow(K, aspect='auto', origin='upper', cmap=P.cmap(), vmin=0, vmax=1,
                       extent=[0, L * M.H_NM * 1e-3, t[-1], 0], interpolation='nearest')
        ax.set_title(TITLE[k], fontsize=7.5, linespacing=1.3)
        ax.set_xticks([0, 5, 10] if j == len(runs) - 1 else [0, 5])   # a '10' beside the next panel's '0' reads as one label
        if j == 0:
            ax.set_ylabel('time (s)')
        else:
            ax.set_yticklabels([])
    fig.text(0.615, YB(0.655), r'position along the microtubule ($\mu$m)', ha='center')
    cax = fig.add_axes(R([0.885, 0.695, 0.014, 0.26]))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'normalized motor density $\rho$')
    LETTER(fig, blk, 'B', 0.30, 0.99,
           'MPD-RDDME can simulate Kip3 traffic jams in microtubule plus ends\n'
           'with volume exclusion and motor sidestepping')
    # C sits in the band between the kymographs' x label and the speed panel's frame, at the far
    # left where that label does not reach.
    # 77 characters is what this row holds: the cartoon occupies everything left of 0.30, and the
    # fuller sentence overhung the page by half an inch.
    LETTER(fig, blk, 'C', 0.30, YB(0.645),
           'Motor slowdown arises emergently from MPD-RDDME simulations of Kip3 transport')

    ax_b = fig.add_axes(R([0.435, 0.435, 0.385, 0.16]))
    for k, (meta, d) in runs:
        v, rho, _ = link_tracers_full(d, window=40, w=10, t_min=A.t_min)
        Bs = bin_speed(v, rho)
        if not len(Bs):
            continue
        v0 = meta['v0_nm_s']
        ax_b.errorbar(Bs[:, 0], Bs[:, 1] / v0, yerr=Bs[:, 2] / v0, fmt='o', ms=3.0,
                      color=cols[k], capsize=1.2)
    ax_b.set_xlabel(r'local motor density $\rho$')
    ax_b.set_ylabel(r'$v / v_0$')
    ax_b.set_xlim(0, 1); ax_b.set_ylim(-0.03, 1.2)
    ax_b.text(0.975, 0.94, r'$t \geq$ %.0f s' % A.t_min, transform=ax_b.transAxes,
              fontsize=8, ha='right', va='top')
    # The key goes inside, at the corner the data leaves empty: above the panel it would sit
    # between the kymographs' x label and the plot frame, with room for neither.
    handles = [Line2D([], [], color=cols[k], marker='o', ms=3.0, lw=1.1, label=KEY[k])
               for k, (_meta, _d) in runs]
    # Upper right, under the time note: the lower-left corner looked empty but the key's last
    # row reaches the points at rho = 0.45, while above rho = 0.55 nothing rises past v/v0 = 0.35.
    ax_b.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 0.87), fontsize=7.5,
                frameon=False, handletextpad=0.5, borderpad=0.2, labelspacing=0.3)

    # ---- D: the configurations ---------------------------------------------------------------
    meta_s, d_s = load(A.snap or A.exc)
    ring = [tuple(int(v) for v in yz) for yz in d_s['ring']]
    # D sits low, so C's x label is not crowded by the block beneath it.
    ax_c = fig.add_axes(R([0.03, 0.005, 0.95, 0.365]))
    draw_time_stack(ax_c, d_s, meta_s['sites'], [float(x) for x in A.times.split(',')],
                    ring, 0.5 * (BODY_HI - BODY_LO + 1))
    LETTER(fig, blk, 'D', 0.012, YB(0.385))

    import jcp_style
    jcp_style.check_width(fig, TEXTW, 'kip3_traffic')
    jcp_style.check_text_overlap(fig, 'kip3_traffic')
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None)
    import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import panel_export
    panel_export.export(fig, 'kip3_traffic')
    print('wrote %s.pdf/.png' % A.out)


if __name__ == '__main__':
    main()
