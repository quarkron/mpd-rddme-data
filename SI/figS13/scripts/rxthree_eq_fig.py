"""SI figure fig:three-eq: the subvolume occupancy distributions of A and B at the end of the ternary relaxation
(tau 20), BD against the volume-excluded and the point-particle lattice (2026-11-05).

This was block b of the main-text fig:three-relax-readdy (main/fig2) until 2026-11-05, when the owner swapped it with
the divergence over time (then this SI figure, now Fig. 2b: main/fig2/scripts/fig2_block_b_kl.py).

    python rxthree_eq_fig.py [--install]

Reads main/fig2's data through its pooling (rxthree_data.prepare: the same arms, seeds and tau grid as the main
figure), writes ../plots/{pdf,png}/rxthree_eq.*; --install copies into the manuscript's figures/ ($MANUSCRIPT_FIGURES).
"""
import argparse, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'main', 'fig2', 'scripts'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402
import jcp_style                                             # noqa: E402
import rxthree_data as RD                                    # noqa: E402

PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)

# Canvas and panel geometry, in inches from the top-left corner, as rxthree_kl_fig.py (FIGURE_STYLE.md section 5).
FW, FH = 7.2, 2.92
AX_Y, AX_W, AX_H = 0.42, 2.70, 2.00
AX_X = (0.85, 4.40)
LET_X = (0.05, 3.62)
YHEAD = 2.0                            # headroom: the key sits above every curve (as Fig. 2)


def marginal(ax, n, Pb, L, pk, i, nmax, nm, arms):
    """One occupancy distribution as main/fig2's marginal(): BD in open circles, each lattice arm over it, one key."""
    ax.plot(n, Pb[i], 'o-', color=RD.REF, mfc='white', mew=1.1, ms=4.5)
    for key, lbl, col, sty in arms:
        ax.plot(n, L[key][pk][i], sty, color=col, mfc=col, mew=0, ms=4.0)
    ax.set_xlim(-0.5, nmax); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xticks(np.arange(0, nmax + 1, 10 if nmax > 20 else 4))
    ax.set_xlabel(r'occupancy $n_\mathrm{%s}$' % nm)
    ax.set_ylabel(r'$P(n_\mathrm{%s})$' % nm)
    ax.tick_params(top=True, right=True)
    top = max(np.max(Pb[i]), max(np.max(L[k][pk][i]) for k, *_ in arms))
    ax.set_ylim(top=YHEAD * top)
    hs = [Line2D([], [], marker='o', ls='-', color=RD.REF, mfc='white', mew=1.1, ms=4.5, lw=1.5, label='BD')]
    for k, lbl, col, sty in arms:
        hs.append(Line2D([], [], marker=sty[0], ls=sty[1:], color=col, mfc=col, mew=0, ms=4.0, lw=1.5, label=lbl))
    ax.legend(handles=hs, loc='upper right', fontsize=jcp_style.TEXT_PT, frameon=False, handlelength=1.8,
              labelspacing=0.25, borderpad=0.1, handletextpad=0.4, borderaxespad=0.2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    D = RD.prepare()
    grid, L, arms = D['grid'], D['L'], D['arms']
    i_eq = len(grid) - 1
    print('  distributions at tau %.3g' % grid[i_eq])
    with plt.rc_context(jcp_style.PUB_RC):
        fig = jcp_style.si_page(FW, FH)
        axs = [jcp_style.pub_ax(fig, x, AX_Y, AX_W, AX_H) for x in AX_X]
        marginal(axs[0], np.arange(RD.NMAX_A + 1), D['PbA'], L, 'PA', i_eq, RD.NMAX_A, 'A', arms)
        marginal(axs[1], np.arange(RD.NMAX_B + 1), D['PbB'], L, 'PB', i_eq, RD.NMAX_B, 'B', arms)
        for x, letter in zip(LET_X, 'ab'):
            jcp_style.pub_letter(fig, letter, x, 0.08)
        jcp_style.pub_check(fig, 'rxthree_eq', si=True); jcp_style.check_text_overlap(fig, 'rxthree_eq')
        for ext in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, ext), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, ext, 'rxthree_eq.' + ext), dpi=300)
        jcp_style.pub_fonts(os.path.join(PLOTS, 'pdf', 'rxthree_eq.pdf'), 'rxthree_eq')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'rxthree_eq')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'rxthree_eq.*')))
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, ext, 'rxthree_eq.' + ext), os.path.join(FIGD, ext, 'rxthree_eq.' + ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
