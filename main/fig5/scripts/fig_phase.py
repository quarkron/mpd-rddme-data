"""fig:phase -- polymer-crowder phase separation from a demixed start (split out of fig:polymer on 2026-10-09).

    python fig_phase.py [--install]      # host venv, TeX Live 2026 on PATH

  a  the model: crowder, bath, a chain with its tip growth k+/k-, free monomers and nucleation, each species' D
     (phase_setup.py, here)
  b  chain length over time and at t = 10 s, beside a (demix_scene3d.lengths)
  c  for t = 0, 1, 10 s: the whole box in 3-D beside a slice through its middle, finite-size over point particles
     (demix_scene3d.snapshots)

The drawing code and the runs stay in main/fig4 (scripts/, data/demix_noneq/slab), which SI/figS18 also imports;
this script only lays the figure out. Writes ../plots/phase.{pdf,png}; --install copies them into the manuscript.
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG4 = os.path.join(HERE, '..', '..', 'fig4', 'scripts')
sys.path.insert(0, FIG4)
import fig_polymer_r2 as F                                               # noqa: E402
import demix_scene3d as D3                                               # noqa: E402
sys.path.insert(0, HERE)
import phase_setup as PS                                                 # noqa: E402
import jcp_style                                                         # noqa: E402  (fig_polymer_r2 put common/ on the path)
import matplotlib                                                        # noqa: E402
import matplotlib.pyplot as plt                                          # noqa: E402

TEXTW, FIGH = 6.5, 5.34
OUT = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)


def AX(x, ytop, w, h):
    """An axes rectangle in inches from the top-left (as fig_polymer_r2.AX)."""
    return [x / TEXTW, 1.0 - (ytop + h) / FIGH, w / TEXTW, h / FIGH]


def YT(ytop):
    return 1.0 - ytop / FIGH


def draw():
    F.FIGH = FIGH                                  # fig_polymer_r2.block places its sentence in figure fractions
    bold = ((lambda t: r'\textbf{%s}' % t) if matplotlib.rcParams.get('text.usetex') else (lambda t: t))
    fig = plt.figure(figsize=(TEXTW, FIGH))
    # ---- a: the model (letter alone), the left half of the first band (phase_setup.py).
    PS.panel_setup(fig.add_axes(AX(0.05, 0.22, 3.30, 2.46)))
    F.block(fig, 0.010, YT(0.20), 'A', None, bold)
    # ---- b: chain length over time and at t = 10 s, the right half of the first band (demix_scene3d.lengths).
    XB = 3.62
    F.block(fig, XB / TEXTW, YT(0.20), 'B', 'Finite-size effects alter', bold)
    jcp_style.block_note(fig, XB / TEXTW, YT(0.20 + F.LETTER_LINE + 0.165), 'polymerization equilibria', size=8.5,
                         color=F.INK)
    D3.lengths(fig.add_axes(AX(4.08, 0.80, 1.02, 1.30)), fig.add_axes(AX(5.55, 0.80, 0.90, 1.30)), F.SLAB_RUNS,
               F.ARMS_D, (F.C_POLY, F.C_CROWD))
    # ---- c: the snapshots across the width.
    F.rule(fig, YT(2.72))
    F.block(fig, 0.010, YT(2.86), 'C', 'Phase separation of polymer-crowder mixture persists in volume-excluded '
            'lattice simulations', bold)
    D3.snapshots(fig, AX, YT, TEXTW, F.SLAB_RUNS, F.ARMS_D, (F.C_POLY, F.C_CROWD), Y0=3.50)
    jcp_style.check_width(fig, TEXTW, 'phase')
    jcp_style.check_text_overlap(fig, 'phase')
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        q = os.path.join(OUT, 'phase.%s' % ext)
        fig.savefig(q, dpi=300); print('wrote', q)
    import panel_export                                  # each panel alone, no letter or headline -> ../plots/panels/
    panel_export.export(fig, 'phase')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    rc, tex = F.style()
    with plt.rc_context(rc):
        draw()
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copy(os.path.join(OUT, 'phase.%s' % ext), os.path.join(FIGD, ext, 'phase.%s' % ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
