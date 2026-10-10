"""SI figure fig:rx-field-kl-time: divergence of the subvolume occupancy distributions from the iPRD reference in time,
for the reversible reaction in the sinusoidal field (main-text fig:rx-field-kl). This was block C of that figure until
2026-10-05; it moved to the supplement unchanged.

    python rx_field_kl_fig.py [--install]

Reads main/fig3's data and loaders (the same runs as the main figure; data/kl3way_cache.npz made by kl3way.py), writes
../plots/{pdf,png}/rx_field_kl.*; --install copies into the manuscript's figures/.
"""
import argparse, os, shutil, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG3 = os.path.join(HERE, '..', '..', '..', 'main', 'fig3', 'scripts')
sys.path.insert(0, FIG3)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import jcp_style                                             # noqa: E402
import rx_field_data as D                                    # noqa: E402
from rx_field_fig import ARMS                                # noqa: E402

TEXTW, H = 6.5, 2.15
PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5); tex = True
    except Exception:
        tex = False
    rc = dict(jcp_style.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if tex else {'mathtext.fontset': 'cm'})
    rc = jcp_style.sans(rc, tex)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(TEXTW, H))
        axs = [fig.add_axes([0.105, 0.22, 0.385, 0.70]), fig.add_axes([0.590, 0.22, 0.385, 0.70])]
        for a, name in zip(axs, ('A', 'C')):
            a.grid(alpha=0.25, lw=0.5, which='both')
            for tag, lbl, col, ls, mk, panels in ARMS:
                if 'kl' not in panels:
                    continue
                xs, ys = D.divergence(name, 'excl' if tag == 'exc' else 'ideal')
                a.plot(xs, ys, mk + ls, color=col, mfc='white' if tag == 'exc' else col, mew=1.1, ms=3.4, lw=1.1,
                       label=lbl)
                print('  %s %-16s D_KL median tau>=1: %.2e' % (name, lbl, np.median(ys[xs >= 1])))
            a.set_xscale('log'); a.set_yscale('log')
            a.set_xlabel(r'$\tau = Dt/h^{2}$', labelpad=1.5)
            a.set_ylim(bottom=2e-5)                   # headroom below the curves for the key
            a.text(0.97, 0.05, r'species $\mathsf{%s}$' % name, transform=a.transAxes, ha='right', va='bottom')
        axs[0].set_ylabel(r'$D_{\mathrm{KL}}$ from iPRD')
        for a in axs:   # the same key in every plot; lower left is empty in both
            a.legend(loc='lower left', fontsize=7.5, frameon=False, handlelength=1.8)
        jcp_style.check_width(fig, TEXTW, 'rx_field_kl'); jcp_style.check_text_overlap(fig, 'rx_field_kl')
        for ext in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, ext), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, ext, 'rx_field_kl.' + ext), dpi=300)
        import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export
        panel_export.export(fig, 'rx_field_kl')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'rx_field_kl.*')), 'TEX=%s' % tex)
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, ext, 'rx_field_kl.' + ext), os.path.join(FIGD, ext, 'rx_field_kl.' + ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
