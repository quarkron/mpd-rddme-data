"""SI figure fig:lp-match: matching the persistence length of the off-lattice (ReaDDy) chain to the lattice chain.

    python lp_match_fig.py [--install]

  a  the two chains: the fcc lattice chain (bonds of sqrt(2) h on one sublattice, discrete turns, bending energy
     kappa (1 + cos theta), theta the joint angle) and the off-lattice bead-spring chain (harmonic bonds, harmonic angle k_theta (theta - pi)^2)
  b  lattice: L_p/<b> against kappa, the exact phantom-chain closed form (common/fcc_exact.lp_over_b) and the lattice
     chain simulated with the paper's move set (main/fig4 data/fcc_si/statics_kappa.npz), one-step estimator
     L_p/<b> = -1/ln<cos theta>
  c  off-lattice: L_p/<b> against k_theta from the converged scan (lp_scan.py: 11 beads, 3.2e6 steps, 8 seeds, coil
     and straight starts), the kappa = 4 target, the matched k_theta, and the value the fig:polymer c reference used
Writes ../plots/{pdf,png}/lp_match.*; --install copies into the manuscript's figures/.
"""
import argparse, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'main', 'fig4', 'scripts'))
import fcc_exact as FX                          # noqa: E402
import fig_polymer_r2 as F4                     # noqa: E402  (house style, palette)
import jcp_style                                # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch   # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
STATICS = os.path.join(HERE, '..', '..', '..', 'main', 'fig4', 'data', 'fcc_si', 'statics_kappa.npz')
PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)
K_USED = 2.3054                 # fig:polymer c reference arms (readdy_crowd_campaign.sh, readdy_chain_bath.py)
KAPPA = 4.0
INK, POLY, REF = F4.INK, F4.C_POLY, F4.KP.ARM['ref']


def scan(n):
    f = os.path.join(DATA, 'lp_scan_n%d.npz' % n)
    return np.load(f) if os.path.exists(f) else None


def joint_angle(ax, a, j, b, r=0.55):
    """Mark theta at joint j: the interior angle between the bonds to the neighbours a and b (pi when straight)."""
    da = np.degrees(np.arctan2(*(a - j)[::-1])); db = np.degrees(np.arctan2(*(b - j)[::-1]))
    t1, t2 = (da, db) if (db - da) % 360 < 180 else (db, da)
    ax.add_patch(Arc(tuple(j), 2 * r, 2 * r, theta1=t1, theta2=t2, color=INK, lw=0.9, zorder=4))
    mid = np.radians(t1 + ((t2 - t1) % 360) / 2)
    ax.text(j[0] + 1.75 * r * np.cos(mid), j[1] + 1.75 * r * np.sin(mid), r'$\theta$', fontsize=8,
            ha='center', va='center', zorder=4)


def panel_a(ax):
    """Left: a lattice chain on a square grid (the 2-D picture of one fcc sublattice: diagonal bonds), a turn marked.
    Right: a bead-spring chain off the lattice, a turn marked."""
    for v in range(0, 6):
        ax.plot([v, v], [0, 4], color=F4.KP.CAGE, lw=0.4, alpha=0.6, zorder=0)
    for v in range(0, 5):
        ax.plot([0, 5], [v, v], color=F4.KP.CAGE, lw=0.4, alpha=0.6, zorder=0)
    lat = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 1.5], [4.5, 2.5], [4.5 - 1, 3.5]])
    ax.plot(lat[:, 0], lat[:, 1], '-', color=INK, lw=1.4, zorder=2)
    ax.plot(lat[:, 0], lat[:, 1], 'o', ms=9, mfc=POLY, mec=INK, mew=0.6, zorder=3)
    joint_angle(ax, lat[1], lat[2], lat[3])
    ax.text(1.05, 1.0, r'$b=\sqrt{2}\,h$', fontsize=7.5, ha='left', va='top', rotation=45)
    ax.text(2.5, 4.35, 'lattice (fcc)', ha='center', fontsize=8)
    ax.text(2.5, -0.45, r'$\kappa\,(1+\cos\theta)$', ha='center', va='top', fontsize=8)
    # off-lattice chain, shifted right
    x0 = 6.6
    pts = np.array([[x0 + 0.2, 0.6], [x0 + 1.25, 1.45], [x0 + 2.45, 1.85], [x0 + 2.85, 3.05], [x0 + 4.05, 3.45]])
    for a_, b_ in zip(pts[:-1], pts[1:]):                       # springs as zigzags
        t = np.linspace(0, 1, 40); d = b_ - a_; nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        z = np.where((t > 0.2) & (t < 0.8), 0.09 * np.sin(2 * np.pi * 5 * (t - 0.2) / 0.6), 0.0)
        p = a_[None] + t[:, None] * d[None] + z[:, None] * nrm[None]
        ax.plot(p[:, 0], p[:, 1], '-', color=INK, lw=0.9, zorder=2)
    ax.plot(pts[:, 0], pts[:, 1], 'o', ms=9, mfc=REF, mec=INK, mew=0.6, zorder=3)
    joint_angle(ax, pts[1], pts[2], pts[3])
    ax.text(x0 + 2.4, 4.35, 'off-lattice (ReaDDy)', ha='center', fontsize=8)
    ax.text(x0 + 2.4, -0.45, r'$k_\theta\,(\theta-\pi)^2$', ha='center', va='top', fontsize=8)
    ax.set_xlim(-0.2, x0 + 5.0); ax.set_ylim(-1.1, 4.7); ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    rc, tex = F4.style()
    with plt.rc_context(rc):
        bold = (lambda t: r'\textbf{%s}' % t) if tex else (lambda t: t)
        fig = plt.figure(figsize=(6.5, 4.6))
        panel_a(fig.add_axes([0.06, 0.60, 0.88, 0.36]))
        fig.text(0.01, 0.985, bold('a'), fontsize=12, va='top')
        # b: lattice
        ab = fig.add_axes([0.09, 0.10, 0.37, 0.40]); fig.text(0.01, 0.55, bold('b'), fontsize=12, va='top')
        kk = np.linspace(0.0, 8.0, 200)
        ab.plot(kk, [FX.lp_over_b(k) for k in kk], '-', color=INK, lw=1.2, label='exact (phantom chain)')
        z = np.load(STATICS, allow_pickle=True)
        rows = sorted([(float(z['%d_kappa' % i]), float(z['%d_mean_cos' % i])) for i in range(int(z['n_rows']))])
        kap = np.array([r[0] for r in rows if r[0] > 0]); lpl = np.array([-1.0 / np.log(r[1]) for r in rows if r[0] > 0])
        ab.plot(kap, lpl, 'o', ms=4.5, mfc=POLY, mec=INK, mew=0.6, label='lattice simulation')
        tgt = float(FX.lp_over_b(KAPPA))
        ab.axhline(tgt, color=INK, lw=0.7, ls='--'); ab.axvline(KAPPA, color=INK, lw=0.7, ls=':')
        ab.set_xlabel(r'bending stiffness $\kappa$ ($k_{\mathrm{B}}T$)'); ab.set_ylabel(r'$L_p/\langle b\rangle$')
        ab.set_xlim(0, 8.2); ab.set_ylim(0, 30); ab.grid(alpha=0.25, lw=0.4)
        ab.legend(loc='upper left', fontsize=7, frameon=False)
        # c: off-lattice scan
        ac = fig.add_axes([0.60, 0.10, 0.37, 0.40]); fig.text(0.50, 0.55, bold('c'), fontsize=12, va='top')
        s11 = scan(11)
        r = s11['rows']; ac.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt='o', ms=4.5, mfc=REF, mec=INK, mew=0.6,
                                     color=INK, elinewidth=0.8, capsize=1.5, label='ReaDDy, 11 beads')
        s32 = scan(32)
        if s32 is not None:
            r32 = s32['rows']; ac.errorbar(r32[:, 0], r32[:, 1], yerr=r32[:, 2], fmt='s', ms=3.6, mfc='white', mec=INK,
                                           mew=0.7, color=INK, elinewidth=0.8, capsize=1.5, label='ReaDDy, 32 beads')
        o = np.argsort(r[:, 1]); kmatch = float(np.exp(np.interp(np.log(tgt), np.log(r[o, 1]), np.log(r[o, 0]))))
        ac.axhline(tgt, color=INK, lw=0.7, ls='--', label=r'lattice target, $\kappa = 4$')
        ac.plot([kmatch], [tgt], '*', ms=10, color=POLY, mec=INK, mew=0.6, label=r'matched $k_\theta = %.2f$' % kmatch)
        i_used = int(np.argmin(abs(r[:, 0] - K_USED)))
        ac.plot([K_USED], [r[i_used, 1]], 'D', ms=6, mfc='none', mec=F4.KP.ARM['exc'], mew=1.3,
                label=r'Fig.\ 4b reference, $k_\theta = %.2f$' % K_USED if tex else 'Fig. 4b reference, k = %.2f' % K_USED)
        ac.set_xlabel(r'angle stiffness $k_\theta$ ($k_{\mathrm{B}}T$)'); ac.set_ylabel(r'$L_p/\langle b\rangle$')
        ac.set_xlim(0, 4.2); ac.set_ylim(0, 9.5)                 # the match region; the scan continues to 8 (printed); ac.grid(alpha=0.25, lw=0.4)
        ac.legend(loc='upper left', fontsize=6.6, frameon=False)
        print('target L_p/<b> %.4f (kappa 4); matched k_theta %.4f; at k_theta %.4f: %.3f +- %.3f (%+.1f %%)'
              % (tgt, kmatch, K_USED, r[i_used, 1], r[i_used, 2], 100 * (r[i_used, 1] / tgt - 1)))
        jcp_style.check_width(fig, 6.5, 'lp_match'); jcp_style.check_text_overlap(fig, 'lp_match')
        for e in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, e), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, e, 'lp_match.' + e), dpi=300)
        import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export
        panel_export.export(fig, 'lp_match')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'lp_match.*')))
    if A.install:
        for e in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, e, 'lp_match.' + e), os.path.join(FIGD, e, 'lp_match.' + e))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
