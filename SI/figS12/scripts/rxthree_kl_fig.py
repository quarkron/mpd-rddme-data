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

TEXTW, H = 6.5, 2.2
PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    D = RD.prepare()
    grid, L, arms = D['grid'], D['L'], D['arms']
    xt = [0, 0.5, 1, 2, 4]
    with plt.rc_context(RD.style(sans=True)):
        fig = plt.figure(figsize=(TEXTW, H))
        axs = [fig.add_axes([0.105, 0.22, 0.385, 0.62]), fig.add_axes([0.590, 0.22, 0.385, 0.62])]
        for a, kk, lab in zip(axs, ('klA', 'klB'), (r'$\mathsf{A}$ (2 nm)', r'$\mathsf{B}$ (4 nm)')):
            for key, lbl, col, sty in arms:
                y = np.asarray(L[key][kk])
                a.plot(grid, y, sty, color=col, mfc=col, mew=0, ms=3.0, label=lbl)
                late = y[grid >= 1.0]
                print('  %s %-16s peak %.3g at tau %.3g; median tau>=1 %.3g; final %.3g'
                      % (kk, lbl, y.max(), grid[np.argmax(y)], np.median(late), y[-1]))
            a.set_xscale('symlog', linthresh=grid[1], linscale=0.5); a.set_yscale('log')
            a.set_xlim(-0.02, grid[-1] * 1.15)
            a.set_xticks(xt); a.set_xticklabels([str(v) for v in xt])
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            a.set_xlabel(r'time $\tau = D_\mathsf{A} t/h^{2}$')
            a.set_title(lab, pad=3, fontsize=8.0); a.grid(alpha=0.25, lw=0.4, which='both')
            lo, hi = a.get_ylim(); a.set_ylim(lo / 8, hi)          # headroom below the curves for the key
            a.legend(loc='lower left', fontsize=7.2, frameon=False, handlelength=2.0)
        axs[0].set_ylabel(r'$D_{\mathrm{KL}}$ from BD')
        jcp_style.check_width(fig, TEXTW, 'rxthree_kl'); jcp_style.check_text_overlap(fig, 'rxthree_kl')
        for ext in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, ext), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, ext, 'rxthree_kl.' + ext), dpi=300)
        import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export
        panel_export.export(fig, 'rxthree_kl')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'rxthree_kl.*')))
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, ext, 'rxthree_kl.' + ext), os.path.join(FIGD, ext, 'rxthree_kl.' + ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
