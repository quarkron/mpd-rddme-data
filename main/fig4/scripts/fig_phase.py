"""fig:phase -- polymer-crowder phase separation from a demixed start (split out of fig:polymer on 2026-10-09).

    python fig_phase.py [--install]      # host venv, TeX Live 2026 on PATH

  a  the model: crowder, bath, a chain with its tip growth k+/k-, free monomers and nucleation (D values in the caption)
     (phase_setup.py, here)
  b  chain length over time and at t = 10 s, beside a (demix_scene3d.lengths_pub)
  c  for t = 0, 1, 10 s: the whole box in 3-D beside a slice through its middle, finite-size over point particles
     (demix_scene3d.snapshots_pub; one row per time)

The drawing code and the runs stay in SI/figS8 (scripts/, data/demix_noneq/slab), which SI/figS19 also imports;
this script only lays the figure out. Writes ../plots/phase.{pdf,png}; --install copies them into the manuscript.
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG4 = os.path.join(HERE, '..', '..', '..', 'SI', 'figS8', 'scripts')   # the chain-figure modules (was main/fig4)
sys.path.insert(0, FIG4)
import fig_polymer_r2 as F                                               # noqa: E402
import demix_scene3d as D3                                               # noqa: E402
sys.path.insert(0, HERE)
import phase_setup as PS                                                 # noqa: E402
import jcp_style                                                         # noqa: E402  (fig_polymer_r2 put common/ on the path)
import matplotlib.pyplot as plt                                          # noqa: E402

OUT = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)


# ---- the publication page (FIGURE_STYLE.md, 2026-10-19): 8.5 x 11 in, Arial, letter + title on one line ---------
J = jcp_style
YA, YC = 0.50, 5.12               # in: header lines of the first row (a, b) and of c
XB = 4.95                         # in: panel b's left edge (letter); its plots start at XBP
XBP, BW = 5.58, 2.30


def head2(fig, letter, line1, line2, x, ytop, w):
    """pub_header with a title too long for its width wrapped onto a second centred line (same weight, gid)."""
    J.pub_header(fig, letter, line1, x, ytop, w)
    y = 1.0 - (ytop + J.LETTER_PT / 72.0 + 0.23) / J.PAGE_H
    t = fig.text((x + 0.5 * w) / J.PAGE_W, y, line2, fontsize=J.TITLE_PT, fontweight='bold', ha='center',
                 va='baseline', color=F.INK)
    t.set_gid('headline')


def draw():
    fig = J.pub_page()
    W = J.PAGE_W - 2 * J.MARGIN
    # ---- a: the model (phase_setup.py), the left of the first row.
    wa = XB - 0.25 - J.MARGIN
    J.pub_header(fig, 'a', 'Polymer assembly among crowders', J.MARGIN, YA, wa)
    PS.panel_setup(J.pub_ax(fig, J.MARGIN, YA + 0.85, wa, wa * 7.1 / 10.4))
    # ---- b: chain length over time and at t = 10 s, stacked at the right of the first row.
    head2(fig, 'b', 'Finite-size effects alter', 'polymerization equilibria', XB, YA, J.PAGE_W - J.MARGIN - XB)
    D3.lengths_pub(J.pub_ax(fig, XBP, YA + 0.95, BW, 1.25), J.pub_ax(fig, XBP, YA + 2.72, BW, 1.25), F.SLAB_RUNS,
                   F.ARMS_D, (F.C_POLY, F.C_CROWD))
    # ---- c: the snapshots, one row per time.
    J.pub_rule(fig, YC - 0.13)
    head2(fig, 'c', 'Phase separation of polymer-crowder mixture persists', 'in volume-excluded lattice simulations',
          J.MARGIN, YC, W)
    D3.snapshots_pub(fig, F.SLAB_RUNS, F.ARMS_D, (F.C_POLY, F.C_CROWD), J.MARGIN, YC + 0.92)
    J.pub_check(fig, 'phase')
    J.check_text_overlap(fig, 'phase')
    os.makedirs(OUT, exist_ok=True)
    q = os.path.join(OUT, 'phase.pdf')
    fig.savefig(q); print('wrote', q)
    png_from_pdf(q)
    J.pub_fonts(q, 'phase')
    import panel_export                                  # each panel alone, no letter or headline -> ../plots/panels/
    for f in panel_export.export(fig, 'phase'):
        if f.endswith('.pdf'):
            png_from_pdf(f)


def png_from_pdf(pdf, dpi=300):
    """The PNG rasterised from the PDF, not by Agg: Agg clips a subscript that holds only a minus sign (k-minus
    in panel a renders as a bare k; matplotlib 3.10 mathtext box depth), while the PDF carries it."""
    import fitz
    fitz.open(pdf)[0].get_pixmap(dpi=dpi).save(pdf[:-4] + '.png')
    print('wrote', pdf[:-4] + '.png')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    with plt.rc_context(J.PUB_RC):
        draw()
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copy(os.path.join(OUT, 'phase.%s' % ext), os.path.join(FIGD, ext, 'phase.%s' % ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
