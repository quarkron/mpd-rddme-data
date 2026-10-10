"""SI Figs. S2 and S3 (wm2_flatness, wm3_intravoxel_offset): the closed-form well-mixing figures.

Extracted 2026-10-01 from harmonic_trap.ipynb (cells 10 and 11, unchanged except the colours, the
WM3 key position and the output paths) so the two closed-form figures can be redrawn without the
notebook, whose first cell imports an engine build that no longer exists. Nothing here simulates:
every curve is an analytic expression of the constants below (cell 1 of the notebook).

Colours follow the main-text palette (kip3_palette): the distance from the well centre is ordered,
so it takes the sequential ramp, and the second-order term the main-text green.

usage (in the lm2.6 container, TeX Live on PATH):  python wm_closedform_figs.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---- physics, as cell 1 of harmonic_trap.ipynb (box units; 1 unit = UNIT meters) ----------------
L      = 256.0
UNIT   = 1e-8
D      = 1e-12
K      = 0.01
SIGMA  = 1.0 / np.sqrt(K)
DK     = D * K / UNIT**2
C_STAB = 0.10
N_SWEEP = [32, 64, 128, 256, 512]

import os
FIG_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots'))   # vex-rddme: plots/{pdf,png}
STOPS = ['#050b23', '#0e2159', '#17509b', '#2093b4', '#41c5bd', '#a3dda0', '#e9f7bb']   # kip3_palette
C_BULK, C_SIG, C_2SIG, C_SECOND = STOPS[1], STOPS[3], STOPS[4], '#9cbf3c'
C_LIM = '#5d7185'                                                                        # INK_DIM

# ======== WM2 (notebook cell 10) ========
# WM2: a priori flatness parameters for the dragged harmonic potential.
#   panel (a) SPATIAL: eps_n^WM = h^n ||D^n phi||.  D^2 phi = k I, D^n phi = 0 (n>=3)
#             => eps_2 = h^2 k (position independent), eps_1(x_v) = h k |x_v - c|.
#   panel (b) TEMPORAL: eps_tau^(n) = tau^n ||d^n phi / dt^n||, for c(t) = c sin(wt)(1,1,1).
#             d_t phi   = -k (x-c).cdot,  cdot = c w cos(wt)(1,1,1)
#             d_t^2 phi =  k |cdot|^2 - k (x-c).cddot
#             At the worst phase for first order (wt = 0: cos = 1, sin = 0) the cddot term
#             drops and d_t^2 phi = 3 k c^2 w^2 exactly -- position independent, the direct
#             analogue of eps_2^WM = h^2 k.  Along the drag diagonal |sum_i(x_i-c_i)| = sqrt(3)|x-c|.
#             Identity: eps_tau^(2) = (eps_tau^(1) at |x-c| = sigma)^2, so second order is
#             automatically satisfied whenever first order is -- the temporal analogue of the
#             spatial expansion terminating at n = 2 for a harmonic well.
# The five markers in BOTH panels are the SAME five runs: the CFL ties tau to h through
# tau = C_STAB * (h*UNIT)^2 / D, so choosing h fixes tau.
# Saves figures/wm2_flatness.{pdf,png}.
import subprocess
from pathlib import Path
from matplotlib.ticker import FuncFormatter, NullFormatter

# The drive parameters panel (b) is evaluated at MUST match Part B (cell 'MovingWell').
C_AMP   = 40.0                     # well-centre oscillation amplitude (box units)
OMEGA   = DK                       # lag 45 deg, attenuation 1/sqrt(2)


try:
    subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
    USE_LATEX = True
except Exception:
    USE_LATEX = False
    print('[WM2] LaTeX binary not found; using matplotlib mathtext fallback.')

mean_abs  = SIGMA * np.sqrt(2.0 / np.pi)          # E|x-c| per axis under N(0, SIGMA^2)
hs_wm     = np.array([L / N for N in N_SWEEP], dtype=float)
eps2      = hs_wm**2 * K
eps1_bulk = hs_wm * K * mean_abs
eps1_sig  = hs_wm * K * SIGMA
eps1_2sig = hs_wm * K * 2 * SIGMA

# tau is not free: the diffusion CFL fixes it once h is chosen.
taus = C_STAB * (hs_wm * UNIT)**2 / D
# |sum_i (x_i - c_i)| = sqrt(3)|x-c| along the drag diagonal; summed coordinate has
# variance 3 SIGMA^2, so its Gaussian mean absolute value is sqrt(3) SIGMA sqrt(2/pi).
pref      = K * C_AMP * OMEGA
et1_bulk  = taus * pref * np.sqrt(3.0) * mean_abs
et1_sig   = taus * pref * np.sqrt(3.0) * SIGMA
et1_2sig  = taus * pref * np.sqrt(3.0) * 2 * SIGMA
et2       = taus**2 * 3.0 * K * C_AMP**2 * OMEGA**2

print('  h    |    tau (s)  |   eps_2^WM  | eps_1^WM(2s) | eps_t1(2s) |  eps_t2')
print('-------+-------------+-------------+--------------+------------+-----------')
for h, t, e2, e1s, t1, t2 in zip(hs_wm, taus, eps2, eps1_2sig, et1_2sig, et2):
    flag = '  *** breakdown ***' if max(e2, e1s, t1, t2) > 1 else ('  <-- marginal' if max(e1s, t1) > 0.3 else '')
    print(f' {h:5.3f} | {t:.3e} |   {e2:.4f}    |    {e1s:.4f}    |   {t1:.4f}   | {t2:.2e}{flag}')
print('identity check  eps_t2 == (eps_t1 at sigma)^2 :',
      np.allclose(et2, et1_sig**2), '  max rel dev %.2e' % np.max(np.abs(et2/et1_sig**2 - 1)))

rc_pub = {
    'font.family': 'serif', 'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 15,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'lines.linewidth': 1.6, 'lines.markersize': 7, 'axes.linewidth': 1.1,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 6.0, 'ytick.major.size': 6.0,
    'xtick.minor.size': 3.0, 'ytick.minor.size': 3.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
}
rc_pub.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']}
              if USE_LATEX else {'mathtext.fontset': 'cm'})

with plt.rc_context(rc_pub):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # ---------------- panel (a): spatial ----------------
    ax.plot(hs_wm, eps1_bulk, 'o-', color=C_BULK, mfc='white', mew=1.4,
            label=r'$\langle \varepsilon^{(1)}_{\mathrm{WM}} \rangle$')
    ax.plot(hs_wm, eps1_sig,  's-', color=C_SIG, mfc='white', mew=1.4,
            label=r'$\varepsilon^{(1)}_{\mathrm{WM}}$ at $|x_v - c| = \sigma$')
    ax.plot(hs_wm, eps1_2sig, '^-', color=C_2SIG, mfc='white', mew=1.4,
            label=r'$\varepsilon^{(1)}_{\mathrm{WM}}$ at $|x_v - c| = 2\sigma$')
    ax.plot(hs_wm, eps2,      'D-', color=C_SECOND, mfc='white', mew=1.4,
            label=r'$\varepsilon^{(2)}_{\mathrm{WM}} = h^2 k$')
    ax.axhline(1.0, color=C_LIM, alpha=0.8, lw=1.0, ls=':')
    ax.text(hs_wm.min(), 0.85, r'$\varepsilon = 1$', color=C_LIM, fontsize=16,
            ha='left', va='top')   # under the line: above it the label left the frame in (b)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks(hs_wm); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$h$', fontsize=20)
    ax.set_ylabel(r'$\varepsilon^{(n)}_{\mathrm{WM}}$', fontsize=20)
    ax.grid(True, which='major', alpha=0.30, lw=0.5)
    ax.grid(True, which='minor', alpha=0.10, lw=0.4)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='0.6', fontsize=14)

    # ---------------- panel (b): temporal ----------------
    bx.plot(taus, et1_bulk, 'o-', color=C_BULK, mfc='white', mew=1.4,
            label=r'$\langle \varepsilon^{(1)}_{\tau} \rangle$')
    bx.plot(taus, et1_sig,  's-', color=C_SIG, mfc='white', mew=1.4,
            label=r'$\varepsilon^{(1)}_{\tau}$ at $|x_v - c| = \sigma$')
    bx.plot(taus, et1_2sig, '^-', color=C_2SIG, mfc='white', mew=1.4,
            label=r'$\varepsilon^{(1)}_{\tau}$ at $|x_v - c| = 2\sigma$')
    bx.plot(taus, et2,      'D-', color=C_SECOND, mfc='white', mew=1.4,
            label=r'$\varepsilon^{(2)}_{\tau} = 3\tau^2 k c^2 \omega^2$')
    bx.axhline(1.0, color=C_LIM, alpha=0.8, lw=1.0, ls=':')
    bx.text(taus.min(), 0.85, r'$\varepsilon = 1$', color=C_LIM, fontsize=16,
            ha='left', va='top')   # under the line: above it the label left the frame in (b)
    bx.set_xscale('log'); bx.set_yscale('log')
    bx.set_xticks(taus)
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v*1e6:g}'))
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.set_xlabel(r'$\tau$ ($\mu$s)', fontsize=20)
    bx.set_ylabel(r'$\varepsilon^{(n)}_{\tau}$', fontsize=20)
    bx.grid(True, which='major', alpha=0.30, lw=0.5)
    bx.grid(True, which='minor', alpha=0.10, lw=0.4)
    bx.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='0.6', fontsize=14)

    for a, tag in ((ax, '(a)'), (bx, '(b)')):
        a.text(0.0, 1.015, tag, transform=a.transAxes, fontsize=17,
               ha='left', va='bottom')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'pdf' / 'wm2_flatness.pdf')
    fig.savefig(FIG_DIR / 'png' / 'wm2_flatness.png', dpi=300)
    print('  -> figures/wm2_flatness.pdf, wm2_flatness.png')


# ======== WM3 (notebook cell 11) ========
# WM3 (closed-form): intra-voxel density at three representative voxels and the
# cell-average vs centroid potential shift Delta(x_v) = (h^2 k / 24)(3 - k|x_v-c|^2).
# LEFT panel: VOXEL-MEAN normalization rho(x_v+delta)/<rho>_vox. Curves do NOT
# pass through 1 at delta=0 in general -- the lift/dip at delta=0 IS the WM
# offset diagnostic, identical to what RIGHT panel shows via Delta(x_v).
# Saves figures/wm3_intravoxel_offset.{pdf,png}.
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import subprocess


# Probe for LaTeX once -- fall back to mathtext if absent.
try:
    subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
    USE_LATEX = True
except Exception:
    USE_LATEX = False
    print('[WM3] LaTeX binary not found; using matplotlib mathtext fallback.')

rc_pub = {
    'font.family':         'serif',
    'font.size':           14,
    'axes.labelsize':      16,
    'axes.titlesize':      15,
    'xtick.labelsize':     14,
    'ytick.labelsize':     14,
    'legend.fontsize':     14,
    'lines.linewidth':     1.6,
    'lines.markersize':    7,
    'axes.linewidth':      1.1,
    'xtick.direction':     'in',  'ytick.direction':     'in',
    'xtick.top':           True,  'ytick.right':         True,
    'xtick.minor.visible': True,  'ytick.minor.visible': True,
    'xtick.major.size':    6.0,   'ytick.major.size':    6.0,
    'xtick.minor.size':    3.0,   'ytick.minor.size':    3.0,
    'xtick.major.width':   1.0,   'ytick.major.width':   1.0,
    'xtick.minor.width':   0.8,   'ytick.minor.width':   0.8,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
}
if USE_LATEX:
    rc_pub.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']})
else:
    rc_pub.update({'mathtext.fontset': 'cm'})

with plt.rc_context(rc_pub):
    fig, a1 = plt.subplots(figsize=(7.5,5.5))

    # (A) intra-voxel rho(x_v+delta)/rho(x_v)  (1-D harmonic Taylor, centroid-normalized)
    for h_demo, ls in ((8.0, '-'), (1.0, '--')):
        dd = np.linspace(-0.5, 0.5, 81) * h_demo            # delta in box units
        for xv, lab, col in [(0.0, r'$x_v=c$', C_BULK), (SIGMA, r'$x_v=c+\sigma$', C_SIG), (2*SIGMA, r'$x_v=c+2\sigma$', C_SECOND)]:
            phip  = K * xv                                   # phi'(x_v)
            phipp = K                                        # phi''
            # Pointwise Taylor (numerator) and closed-form voxel-mean (denominator).
            rho_num   = 1 - phip*dd - 0.5*phipp*dd**2 + 0.5*(phip*dd)**2
            rho_denom = 1 - (h_demo**2 * K / 24.0) * (1.0 - K * xv**2)   # <rho_t>_vox, see eq:avg1d
            rho_ratio = rho_num / rho_denom                               # rho_t(delta)/<rho_t>_vox
            a1.plot(dd/h_demo, rho_ratio, ls=ls, color=col, label=f'$h={h_demo:g}$, {lab}')
    a1.axhline(1.0, color='k', alpha=0.3, lw=0.7)
    a1.set_xlabel(r'$\delta / h$', fontsize=20); a1.set_ylabel(r'$\rho(x_v+\delta)/\langle\rho\rangle_{\rm vox}$', fontsize=20)
    #a1.set_title('Intra-voxel density shape (1-D Taylor, harmonic)')
    a1.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    a1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # Upper right holds no curve: for delta/h > 0 every curve stays below 1.03.
    a1.legend(fontsize=13, ncol=2, loc='upper right', frameon=True, framealpha=0.95, edgecolor='0.6')
    a1.grid(True, alpha=0.3)

    # (B) Delta(x_v) = (h^2 k / 24)(3 - k|x_v-c|^2), 1-D slice through center
    # xv_grid = np.linspace(-3*SIGMA, 3*SIGMA, 400)
    # for h in hs_wm:
    #     a2.plot(xv_grid, (h**2 * K / 24.0) * (3.0 - K*xv_grid**2), label=f'h = {h:g}')
    # xstar = np.sqrt(3.0 / K)                                  # sign-change locus
    # a2.axvline( xstar, color='k', alpha=0.3, ls='--'); a2.axvline(-xstar, color='k', alpha=0.3, ls='--')
    # a2.axhline(0, color='k', alpha=0.3, lw=0.7)
    # a2.text(xstar*1.03, a2.get_ylim()[1]*0.55, fr'$|x-c|=\sqrt{{3/k}}\approx{xstar:.1f}$', fontsize=8)
    # a2.set_xlabel('x - c (box units)'); a2.set_ylabel(r'$\Delta(x_v)=\phi_{\rm avg}-\phi_v$')
    # a2.set_title(r'Cell-average vs centroid potential shift  $(h^2 k/24)(3-k|x-c|^2)$  $\leftarrow$ WM error lives here')
    # a2.legend(fontsize=8); a2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'pdf' / 'wm3_intravoxel_offset.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'png' / 'wm3_intravoxel_offset.png', bbox_inches='tight', dpi=300)
    print('  -> figures/wm3_intravoxel_offset.{pdf,png}')
    plt.show()