"""SI figure: box-wide composition of the sinusoidal-field reaction as it relaxes.

    python rx_field_counts_fig.py [--out out/fig_rx_field_counts] [--install]

The main-text figure (rx_field_fig.py) shows where the particles sit; this one shows how many of
each there are. Two panels, the reactant and the product, each the whole-box count normalized by
the initial reactant count N_A(0), against tau = D t / h^2 for the three descriptions. The shaded
band is the late-time window (tau >= TAU_SS) whose mean the main text quotes, and the build prints
those means so the quoted numbers come from the thing that drew the figure.

Reads the same runs as the main-text figure through rx_field_data.counts_all(). Drawn in the SI
publication style (FIGURE_STYLE.md section 5: Arial, letters a and b, no headline).
"""
import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))   # vex-rddme/common

import jcp_style
import kip3_palette as P
import rx_field_data as D

ARMS = (('ref', 'iPRD', '-', ''),
        ('exc', 'MPD-RDDME', '-', 'o'),
        ('pt', 'point particles', '--', 's'))

# Canvas and panel geometry, in inches from the top-left corner (FIGURE_STYLE.md section 5).
FW, FH = 7.2, 2.92
AX_Y, AX_W, AX_H = 0.42, 2.70, 2.00
AX_X = (0.85, 4.40)                    # left edges of panels a and b
LET_X = (0.05, 3.62)                   # their letters, left of each panel's y label


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default=os.path.join(HERE, '..', 'plots', 'fig_rx_field_counts'))
    p.add_argument('--install', action='store_true',
                   help='also copy into the manuscript figure directories')
    A = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    J = jcp_style

    CNT = D.counts_all()
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(FW, FH)
        axes = [J.pub_ax(fig, x, AX_Y, AX_W, AX_H) for x in AX_X]
        for ax, (name, col, ylab) in zip(axes, (('A', 0, r'$N_\mathrm{A}/N_\mathrm{A}(0)$'),
                                               ('C', 2, r'$N_\mathrm{C}/N_\mathrm{A}(0)$'))):
            ax.axvspan(D.TAU_SS, 20.0, color='0.92', lw=0, zorder=0)
            for tag, lbl, ls, mk in ARMS:
                t, c = CNT[tag]
                m = t > 0                                  # the log axis cannot show t = 0
                kw = dict(color=P.ARM[tag], ls=ls, label=lbl, zorder=2 if tag == 'ref' else 3)
                if mk:
                    kw.update(marker=mk, ms=4.0, mew=1.0, lw=1.2, mfc='white' if tag == 'exc' else P.ARM[tag])
                ax.plot(t[m], c[m, col], **kw)
            ax.set_xscale('log')   # three observation windows at dt 1e-9..1e-7 s span four decades of tau
            ax.set_xlim(3e-3, 20.0)
            ax.set_xlabel(r'time $\tau$')
            ax.set_ylabel(ylab)
            ax.tick_params(which='both', top=True, right=True)
            ax.grid(alpha=0.25, lw=0.5, which='major')
            # The name in the plot's empty corner: above the curves for the reactant, below them for the product.
            yn, va = (0.96, 'top') if name == 'A' else (0.04, 'bottom')
            ax.text(0.03, yn, 'species %s' % name, transform=ax.transAxes, ha='left', va=va, fontsize=J.LABEL_PT)
        # The key goes in the empty middle of the product plot, between the two families of curves.
        axes[1].legend(loc='center right', bbox_to_anchor=(1.0, 0.42))
        for x, letter in zip(LET_X, 'ab'):
            J.pub_letter(fig, letter, x, 0.08)

        os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
        J.pub_check(fig, 'rx_field_counts', si=True)
        J.check_text_overlap(fig, 'rx_field_counts')
        for ext in ('pdf', 'png'):
            fig.savefig('%s.%s' % (A.out, ext), dpi=300)
        J.pub_fonts(A.out + '.pdf', 'rx_field_counts')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'rx_field_counts')
    print('wrote %s.pdf/.png' % A.out)
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
