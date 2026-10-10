"""SI figure fig:demix-starts: polymer-crowder demixing from two other starts than the main-text slab (fig:polymer d).

    python demix_starts_fig.py [--install]

  a  the UNIFORM start (monomers and crowders placed at random; the block d of fig:polymer until 2026-10-09) at
     t = 0, 2, 10 s, and its chain-length panels
  b  the DROPLET start (monomers in a central sphere of half the box volume, crowders outside; demix_slab.sh
     --start droplet) at t = 0, 1, 10 s, and its chain-length panels
Both as fig:phase c draws the slab start: the whole box in 3-D beside a two-tone slice through its middle
(demix_scene3d.snapshots), no headlines (2026-10-09).
Reads main/fig4's data and scripts (data -> fig4). Writes ../plots/{pdf,png}/demix_starts.*; --install copies into
the manuscript's figures/.
"""
import argparse, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
F4 = os.path.join(HERE, '..', '..', '..', 'main', 'fig4', 'scripts')
sys.path.insert(0, F4)
import fig_polymer_r2 as F                      # noqa: E402
import demix_lengths as DL                      # noqa: E402
import demix_scene3d as D3                      # noqa: E402
import jcp_style                                # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402

TEXTW, H = 6.5, 7.92
PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)
DROPLET = os.path.join(F4, '..', 'data', 'demix_noneq', 'droplet')


def AX(x, ytop, w, h):
    return [x / TEXTW, 1.0 - (ytop + h) / H, w / TEXTW, h / H]


def YT(ytop):
    return 1.0 - ytop / H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    rc, tex = F.style()
    with plt.rc_context(rc):
        bold = (lambda t: r'\textbf{%s}' % t) if tex else (lambda t: t)
        fig = plt.figure(figsize=(TEXTW, H))
        cols = (F.C_POLY, F.C_CROWD)
        # Each start as fig:phase c draws the slab: the 3-D box beside its slice at three times (demix_scene3d), and
        # beneath, the chain length over time and its distribution at t = 10 s (demix_lengths). No headlines.
        for letter, top, root, times in (('a', 0.0, F.DEMIX, (0, 2, 10)), ('b', 4.02, DROPLET, (0, 1, 10))):
            fig.text(0.010, 1.0 - (top + 0.20) / H, bold(letter), fontsize=12, va='baseline')
            D3.snapshots(fig, AX, YT, TEXTW, root, F.ARMS_D, cols, Y0=top + 0.52, times=times)
            DL.panel_lengths(fig.add_axes(AX(0.66, top + 2.56, 2.46, 0.95)), fig.add_axes(AX(3.96, top + 2.56, 2.46, 0.95)),
                             F.ARMS_D, root, (F.C_CROWD, F.C_POLY), ('-', '--'))
        fig.add_artist(plt.Line2D([0.012, 0.988], [YT(3.94)] * 2, transform=fig.transFigure, color=F.INK, lw=0.5,
                                  alpha=0.3))
        jcp_style.check_width(fig, TEXTW, 'demix_starts'); jcp_style.check_text_overlap(fig, 'demix_starts')
        for e in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, e), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, e, 'demix_starts.' + e), dpi=300)
        import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export
        panel_export.export(fig, 'demix_starts')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'demix_starts.*')))
    if A.install:
        for e in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, e, 'demix_starts.' + e), os.path.join(FIGD, e, 'demix_starts.' + e))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
