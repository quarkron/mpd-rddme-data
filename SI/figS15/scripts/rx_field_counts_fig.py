"""SI figure: box-wide composition of the sinusoidal-field reaction as it relaxes.

    python rx_field_counts_fig.py [--out out/fig_rx_field_counts] [--install]

The main-text figure (rx_field_fig.py) shows where the particles sit; this one shows how many of
each there are. Two panels, the reactant and the product, each the whole-box count normalized by
the initial reactant count N_A(0), against tau = D t / h^2 for the three descriptions. The shaded
band is the late-time window (tau >= TAU_SS) whose mean the main text quotes, and the build prints
those means so the quoted numbers come from the thing that drew the figure.

Reads the same runs as the main-text figure through rx_field_data.counts_all(). Runs in the lm2.6
container with TeX Live on PATH.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))   # vex-rddme/common

import jcp_style
import kip3_palette as P
import rx_field_data as D

TEXTW, TEXTH = 6.5, 2.45          # the preprint text width, as rx_field_fig.py
ARMS = (('ref', 'iPRD', '-', '', 1.3),
        ('exc', 'MPD-RDDME', '-', 'o', 0.9),
        ('pt', 'point particles', '--', 's', 0.9))


def has_tex():
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default=os.path.join(HERE, '..', 'plots', 'fig_rx_field_counts'))
    p.add_argument('--install', action='store_true',
                   help='also copy into the manuscript figure directories')
    A = p.parse_args()

    tex = has_tex()
    rc = dict(jcp_style.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True,
               'ytick.right': True})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if tex
              else {'mathtext.fontset': 'cm'})
    rc = jcp_style.sans(rc, tex)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update(rc)

    CNT = D.counts_all()
    fig = plt.figure(figsize=(TEXTW, TEXTH))
    axes = [fig.add_axes([0.085, 0.20, 0.39, 0.62]), fig.add_axes([0.595, 0.20, 0.39, 0.62])]
    for ax, (name, col, ylab) in zip(axes, (('A', 0, r'$N_\mathsf{A}/N_\mathsf{A}(0)$'),
                                           ('C', 2, r'$N_\mathsf{C}/N_\mathsf{A}(0)$'))):
        ax.axvspan(D.TAU_SS, 20.0, color='0.92', lw=0, zorder=0)
        for tag, lbl, ls, mk, lw in ARMS:
            t, c = CNT[tag]
            m = t > 0                                  # the log axis cannot show t = 0
            kw = dict(color=P.ARM[tag], lw=lw, ls=ls, label=lbl, zorder=2 if tag == 'ref' else 3)
            if mk:
                kw.update(marker=mk, ms=3.0, mew=0.9, mfc='white' if tag == 'exc' else P.ARM[tag])
            ax.plot(t[m], c[m, col], **kw)
        ax.set_xscale('log')       # three observation windows at dt 1e-9..1e-7 s span four decades of tau
        ax.set_xlim(3e-3, 20.0)
        ax.set_xlabel(r'$\tau = Dt/h^{2}$')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.5, which='major')
        # One block, one conclusion: the two plots are named by their species, as in the main-text
        # figure, so no inner letters. The name sits above the frame, clear of every curve.
        ax.set_title(r'species $\mathsf{%s}$' % name, fontsize=8, pad=3)
    # The key goes in the empty middle of the product plot, between the two families of curves.
    axes[1].legend(frameon=False, fontsize=7.5, loc='center right', bbox_to_anchor=(1.0, 0.42),
                   handlelength=2.0)
    jcp_style.block_note(fig, 0.010, 0.975, r'Volume exclusion shifts the composition toward the product')

    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    jcp_style.check_width(fig, TEXTW, 'rx_field_counts')
    jcp_style.check_text_overlap(fig, 'rx_field_counts')
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), dpi=300)
    import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import panel_export
    panel_export.export(fig, 'rx_field_counts')
    print('wrote %s.pdf/.png  TEX=%s' % (A.out, tex))
    for tag in ('ref', 'exc', 'pt'):
        t, c = CNT[tag]
        m = t >= D.TAU_SS
        print('  late counts  %-4s  N_A/N_A(0) %.4f   N_C/N_A(0) %.4f   (%d frames)'
              % (tag, c[m, 0].mean(), c[m, 2].mean(), m.sum()))

    if A.install:
        figd = os.environ.get('MANUSCRIPT_FIGURES', '')
        for ext, sub in (('pdf', 'pdf'), ('png', 'png')):
            dst = os.path.join(figd, sub, 'rx_field_counts.' + ext)
            shutil.copyfile('%s.%s' % (A.out, ext), dst)
            print('installed', os.path.normpath(dst))


if __name__ == '__main__':
    main()
