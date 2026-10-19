"""SI figure fig:rx-field-kl-time: divergence of the subvolume occupancy distributions from the iPRD reference in time,
for the reversible reaction in the sinusoidal field (main-text fig:rx-field-kl). This was block C of that figure until
2026-10-05; it moved to the supplement unchanged.

    python rx_field_kl_fig.py [--install]

Reads main/fig3's data and loaders (the same runs as the main figure; data/kl3way_cache.npz made by kl3way.py), writes
../plots/{pdf,png}/rx_field_kl.*; --install copies into the manuscript's figures/.
"""
import argparse, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG3 = os.path.join(HERE, '..', '..', '..', 'main', 'fig3', 'scripts')
sys.path.insert(0, FIG3)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import jcp_style as J                                       # noqa: E402
import rx_field_data as D                                    # noqa: E402
from rx_field_fig import ARMS                                # noqa: E402

PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)


# Canvas and panel geometry, in inches from the top-left corner (FIGURE_STYLE.md section 5).
FW, FH = 7.2, 2.92
AX_Y, AX_W, AX_H = 0.42, 2.70, 2.00
AX_X = (0.85, 4.40)                    # left edges of panels a and b
LET_X = (0.05, 3.62)                   # their letters, left of each panel's y label


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(FW, FH)
        axs = [J.pub_ax(fig, x, AX_Y, AX_W, AX_H) for x in AX_X]
        for a, name in zip(axs, ('A', 'C')):
            a.grid(alpha=0.25, lw=0.5, which='both')
            a.tick_params(which='both', top=True, right=True)
            for tag, lbl, col, ls, mk, panels in ARMS:
                if 'kl' not in panels:
                    continue
                xs, ys = D.divergence(name, 'excl' if tag == 'exc' else 'ideal')
                a.plot(xs, ys, mk + ls, color=col, mfc='white' if tag == 'exc' else col, mew=1.2, ms=5.0,
                       label=lbl)
                print('  %s %-16s D_KL median tau>=1: %.2e' % (name, lbl, np.median(ys[xs >= 1])))
            a.set_xscale('log'); a.set_yscale('log')
            a.set_xlabel(r'time $\tau$')
            a.set_ylim(bottom=2e-5)                   # headroom below the curves for the key
            a.text(0.97, 0.05, 'species %s' % name, transform=a.transAxes, ha='right', va='bottom',
                   fontsize=J.LABEL_PT)
            a.set_ylabel(r'$D_{\mathrm{KL}}$ from iPRD')
        for a in axs:   # the same key in every plot; lower left is empty in both
            a.legend(loc='lower left')
        for x, letter in zip(LET_X, 'ab'):
            J.pub_letter(fig, letter, x, 0.08)
        J.pub_check(fig, 'rx_field_kl', si=True); J.check_text_overlap(fig, 'rx_field_kl')
        for ext in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, ext), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, ext, 'rx_field_kl.' + ext), dpi=300)
        J.pub_fonts(os.path.join(PLOTS, 'pdf', 'rx_field_kl.pdf'), 'rx_field_kl')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'rx_field_kl')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'rx_field_kl.*')))
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, ext, 'rx_field_kl.' + ext), os.path.join(FIGD, ext, 'rx_field_kl.' + ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
