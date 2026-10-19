"""SI figure fig:three-kl: divergence of the subvolume occupancy distributions of the ternary mixture (main-text
fig:three-relax-readdy) from the Brownian reference over the relaxation. This was block C of that figure until
2026-10-05; it moved to the supplement with its discussion.

    python rxthree_kl_fig.py [--install]

Reads main/fig2's data through its own pooling (rxthree_data.prepare: the same arms, seeds and common tau grid as the
main figure), writes ../plots/{pdf,png}/rxthree_kl.*; --install copies into the manuscript's figures/. Prints the
numbers the SI text quotes.
"""
import argparse, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'main', 'fig2', 'scripts'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import jcp_style                                             # noqa: E402
import rxthree_data as RD                                    # noqa: E402

PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)


# Canvas and panel geometry, in inches from the top-left corner (FIGURE_STYLE.md section 5).
FW, FH = 7.2, 2.92
AX_Y, AX_W, AX_H = 0.42, 2.70, 2.00
AX_X = (0.85, 4.40)                    # left edges of panels a and b
LET_X = (0.05, 3.62)                   # their letters, left of each panel's y label


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    D = RD.prepare()
    grid, L, arms = D['grid'], D['L'], D['arms']
    xt = [0, 0.5, 1, 2, 4] + [v for v in (10, 20) if v <= grid[-1] * 1.1]   # the run now reaches tau 20
    with plt.rc_context(jcp_style.PUB_RC):
        fig = jcp_style.si_page(FW, FH)
        axs = [jcp_style.pub_ax(fig, x, AX_Y, AX_W, AX_H) for x in AX_X]
        for a, kk, name in zip(axs, ('klA', 'klB'), 'AB'):
            for key, lbl, col, sty in arms:
                y = np.asarray(L[key][kk])
                a.plot(grid, y, sty, color=col, mfc=col, mew=0, ms=4.0, label=lbl)
                late = y[grid >= 1.0]
                print('  %s %-16s peak %.3g at tau %.3g; median tau>=1 %.3g; final %.3g'
                      % (kk, lbl, y.max(), grid[np.argmax(y)], np.median(late), y[-1]))
            a.set_xscale('symlog', linthresh=grid[1], linscale=0.5); a.set_yscale('log')
            a.set_xlim(-0.02, grid[-1] * 1.15)
            a.set_xticks(xt); a.set_xticklabels([str(v) for v in xt])
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            a.tick_params(which='both', top=True, right=True)
            a.set_xlabel(r'time $\tau$')
            a.set_ylabel(r'$D_{\mathrm{KL}}$ from BD')
            a.grid(alpha=0.25, lw=0.5, which='both')
            lo, hi = a.get_ylim(); a.set_ylim(lo / 12, hi)         # headroom below the curves for the key
            a.text(0.97, 0.05, 'species %s' % name, transform=a.transAxes, ha='right', va='bottom',
                   fontsize=jcp_style.LABEL_PT)
            a.legend(loc='lower left')
        for x, letter in zip(LET_X, 'ab'):
            jcp_style.pub_letter(fig, letter, x, 0.08)
        jcp_style.pub_check(fig, 'rxthree_kl', si=True); jcp_style.check_text_overlap(fig, 'rxthree_kl')
        for ext in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, ext), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, ext, 'rxthree_kl.' + ext), dpi=300)
        jcp_style.pub_fonts(os.path.join(PLOTS, 'pdf', 'rxthree_kl.pdf'), 'rxthree_kl')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'rxthree_kl')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'rxthree_kl.*')))
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, ext, 'rxthree_kl.' + ext), os.path.join(FIGD, ext, 'rxthree_kl.' + ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
