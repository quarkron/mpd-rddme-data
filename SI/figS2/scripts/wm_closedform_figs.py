"""SI Figs. S2 and S4a (wm2_flatness, wm3_intravoxel_offset): the closed-form well-mixing figures.

Extracted 2026-10-01 from harmonic_trap.ipynb (cells 10 and 11) so the two closed-form figures can be redrawn
without the notebook, whose first cell imports an engine build that no longer exists. Nothing here simulates: every
curve is an analytic expression of the constants below (cell 1 of the notebook).

2026-10-19: redrawn at final size in the publication style (FIGURE_STYLE.md section 5, common/jcp_style.py PUB_RC):
Arial text and math, 10/11 pt, bold 15 pt letters, no equations or parameter values inside the panels (the
definitions, the unity line and the drive parameters are the caption's). draw_wm3(ax) is imported by
SI/figS4/scripts/wm_sim_figs.py for the combined fig:wm-intravoxel (plots/wm_intravoxel).

Colours follow the main-text palette (kip3_palette): the distance from the well centre is ordered, so it takes the
sequential ramp, and the second-order term the main-text green.

usage:  python wm_closedform_figs.py      # -> ../plots/{pdf,png}/wm2_flatness.*, wm3_intravoxel_offset.*
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import jcp_style as J                               # noqa: E402

# ---- physics, as cell 1 of harmonic_trap.ipynb (box units; 1 unit = UNIT meters) ----------------
L      = 256.0
UNIT   = 1e-8
D      = 1e-12
K      = 0.01
SIGMA  = 1.0 / np.sqrt(K)
DK     = D * K / UNIT**2
C_STAB = 0.10
N_SWEEP = [32, 64, 128, 256, 512]
# The drive parameters panel (b) is evaluated at MUST match Part B (cell 'MovingWell').
C_AMP   = 40.0                     # well-centre oscillation amplitude (box units)
OMEGA   = DK                       # lag 45 deg, attenuation 1/sqrt(2)

FIG_DIR = Path(os.path.join(HERE, '..', 'plots'))   # vex-rddme: plots/{pdf,png}
STOPS = ['#050b23', '#0e2159', '#17509b', '#2093b4', '#41c5bd', '#a3dda0', '#e9f7bb']   # kip3_palette
C_BULK, C_SIG, C_2SIG, C_SECOND = STOPS[1], STOPS[3], STOPS[4], '#9cbf3c'
C_LIM = '#5d7185'                                                                        # INK_DIM


def save(fig, stem):
    """Write ../plots/pdf/<stem>.pdf and ../plots/png/<stem>.png and run the publication checks."""
    pdf = FIG_DIR / 'pdf' / (stem + '.pdf')
    fig.savefig(pdf)
    fig.savefig(FIG_DIR / 'png' / (stem + '.png'), dpi=300)
    J.pub_check(fig, stem, si=True)
    J.check_text_overlap(fig, stem)
    J.pub_fonts(str(pdf), stem)
    print('  -> %s.{pdf,png} (%.2f x %.2f in)' % (stem, fig.get_figwidth(), fig.get_figheight()))


# ======== WM2 (notebook cell 10) ========
# WM2: a priori flatness parameters for the dragged harmonic potential.
#   panel (a) SPATIAL: eps_n^WM = h^n ||D^n phi||.  D^2 phi = k I, D^n phi = 0 (n>=3)
#             => eps_2 = h^2 k (position independent), eps_1(x_v) = h k |x_v - c|.
#   panel (b) TEMPORAL: eps_tau^(n) = tau^n ||d^n phi / dt^n||, for c(t) = c sin(wt)(1,1,1).
#             At the worst phase for first order (wt = 0) d_t^2 phi = 3 k c^2 w^2 exactly -- position
#             independent, the direct analogue of eps_2^WM = h^2 k. Along the drag diagonal
#             |sum_i(x_i-c_i)| = sqrt(3)|x-c|. Identity: eps_tau^(2) = (eps_tau^(1) at |x-c| = sigma)^2.
# The five markers in BOTH panels are the SAME five runs: the CFL ties tau to h through
# tau = C_STAB * (h*UNIT)^2 / D, so choosing h fixes tau.
def flatness():
    mean_abs = SIGMA * np.sqrt(2.0 / np.pi)          # E|x-c| per axis under N(0, SIGMA^2)
    hs = np.array([L / N for N in N_SWEEP], dtype=float)
    taus = C_STAB * (hs * UNIT)**2 / D               # tau is not free: the diffusion CFL fixes it once h is chosen
    pref = K * C_AMP * OMEGA
    f = dict(hs=hs, taus=taus,
             eps1_bulk=hs * K * mean_abs, eps1_sig=hs * K * SIGMA, eps1_2sig=hs * K * 2 * SIGMA, eps2=hs**2 * K,
             et1_bulk=taus * pref * np.sqrt(3.0) * mean_abs, et1_sig=taus * pref * np.sqrt(3.0) * SIGMA,
             et1_2sig=taus * pref * np.sqrt(3.0) * 2 * SIGMA, et2=taus**2 * 3.0 * K * C_AMP**2 * OMEGA**2)
    print('  h    |    tau (s)  |   eps_2^WM  | eps_1^WM(2s) | eps_t1(2s) |  eps_t2')
    for h, t, e2, e1s, t1, t2 in zip(hs, taus, f['eps2'], f['eps1_2sig'], f['et1_2sig'], f['et2']):
        print(f' {h:5.3f} | {t:.3e} |   {e2:.4f}    |    {e1s:.4f}    |   {t1:.4f}   | {t2:.2e}')
    print('identity check  eps_t2 == (eps_t1 at sigma)^2 :', np.allclose(f['et2'], f['et1_sig']**2))
    return f


SERIES = (('bulk', 'o', C_BULK, 'first order, averaged'),
          ('sig', 's', C_SIG, r'first order at $\sigma$'),
          ('2sig', '^', C_2SIG, r'first order at $2\sigma$'),
          ('second', 'D', C_SECOND, 'second order'))


def _flat_panel(a, x, ys, xlabel, ylabel, xfmt):
    for (_, m, c, lab), y in zip(SERIES, ys):
        a.plot(x, y, m + '-', color=c, mfc='white', mew=1.2, label=lab)
    a.axhline(1.0, color=C_LIM, alpha=0.8, lw=1.0, ls=':')       # unity: the caption says so
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xticks(x)
    a.xaxis.set_major_formatter(FuncFormatter(xfmt))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlim(x.min() / 1.4, x.max() * 1.4)
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.grid(True, which='major', alpha=0.25, lw=0.5)


def wm2_figure():
    f = flatness()
    W, H = 6.5, 3.42
    XA, XB, AW, AH, YT = 0.66, 3.96, 2.44, 2.28, 0.36
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(W, H)
        ax = J.pub_ax(fig, XA, YT, AW, AH)
        bx = J.pub_ax(fig, XB, YT, AW, AH)
        _flat_panel(ax, f['hs'], (f['eps1_bulk'], f['eps1_sig'], f['eps1_2sig'], f['eps2']),
                    r'lattice spacing $h$', r'spatial flatness $\varepsilon^{(n)}_{\mathrm{WM}}$',
                    lambda v, _: f'{v:g}')
        _flat_panel(bx, f['taus'], (f['et1_bulk'], f['et1_sig'], f['et1_2sig'], f['et2']),
                    'timestep $\\tau$ (μs)', r'temporal flatness $\varepsilon^{(n)}_{\tau}$',
                    lambda v, _: f'{v * 1e6:g}')
        ax.set_ylim(1.2e-3, 3.0)
        bx.set_ylim(1.2e-6, 3.0)
        # one key for both panels, below them: inside either frame it would sit on the second-order curve
        hs, ls = ax.get_legend_handles_labels()
        fig.legend(hs, ls, loc='lower center', bbox_to_anchor=(0.5 * (XA + XB + AW) / W, 0.0), ncol=4,
                   columnspacing=1.3, handlelength=1.8, handletextpad=0.5)
        J.pub_letter(fig, 'a', 0.02, 0.04)
        J.pub_letter(fig, 'b', XB - 0.64, 0.04)
        save(fig, 'wm2_flatness')
    return fig


# ======== WM3 (notebook cell 11) ========
# WM3 (closed-form): intra-voxel density at three representative voxels, VOXEL-MEAN normalized
# rho(x_v+delta)/<rho>_vox. Curves do NOT pass through 1 at delta=0 in general -- the lift/dip at delta=0 IS the
# WM offset diagnostic, Delta(x_v) = (h^2 k / 24)(3 - k|x_v-c|^2).
WM3_POS = ((0.0, 'at the center', C_BULK), (SIGMA, r'$\sigma$ from the center', C_SIG),
           (2 * SIGMA, r'$2\sigma$ from the center', C_SECOND))
WM3_H = ((8.0, '-'), (1.0, '--'))


def draw_wm3(a):
    """Panel a of fig:wm-intravoxel: the closed-form intra-voxel density at h = 8 (solid) and h = 1 (dashed)."""
    for h_demo, ls in WM3_H:
        dd = np.linspace(-0.5, 0.5, 81) * h_demo            # delta in box units
        for xv, _, col in WM3_POS:
            phip, phipp = K * xv, K                            # phi'(x_v), phi''
            rho_num = 1 - phip * dd - 0.5 * phipp * dd**2 + 0.5 * (phip * dd)**2
            rho_denom = 1 - (h_demo**2 * K / 24.0) * (1.0 - K * xv**2)   # <rho_t>_vox, see eq:avg1d
            a.plot(dd / h_demo, rho_num / rho_denom, ls=ls, color=col)
    a.axhline(1.0, color=C_LIM, alpha=0.5, lw=0.7)
    a.set_xlabel(r'offset in the subvolume $\delta / h$')
    a.set_ylabel(r'$\rho(x_v+\delta)$ / subvolume mean')
    a.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
    a.yaxis.set_major_locator(MaxNLocator(nbins=5))
    a.grid(True, alpha=0.25, lw=0.5)
    pos = a.legend([Line2D([], [], color=c) for _, _, c in WM3_POS], [l for _, l, _ in WM3_POS],
                   loc='upper right')
    a.add_artist(pos)
    a.legend([Line2D([], [], color=C_LIM, ls=ls) for _, ls in WM3_H], [r'$h = %g$' % h for h, _ in WM3_H],
             loc='lower left')


PANEL_W, PANEL_H = 3.2, 2.9        # in, one fig:wm-intravoxel panel on its own file
PANEL_AX = (0.62, 0.12, 2.5, 2.28)  # x, ytop, w, h of its axes


def wm3_figure():
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(PANEL_W, PANEL_H)
        draw_wm3(J.pub_ax(fig, *PANEL_AX))
        save(fig, 'wm3_intravoxel_offset')
    return fig


def main():
    fig = wm2_figure()
    if os.path.basename(os.path.dirname(HERE)) == 'figS2':   # the S4 copy writes wm2 too, but S2 owns its panels
        import panel_export
        panel_export.export(fig, 'wm2_flatness')
    wm3_figure()


if __name__ == '__main__':
    main()
