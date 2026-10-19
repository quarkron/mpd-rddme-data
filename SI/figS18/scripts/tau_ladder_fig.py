"""SI figure fig:tau-ladder: the timestep error of the operator splitting, fixed against randomized step order.

    python tau_ladder_fig.py [--install]

Reads ../data (copies of the runs; see ../README.md), writes ../plots/{pdf,png}/tau_ladder.*. Two panels, linear axes:

  a  reversible binding in the sinusoidal field (the Fig. 3 system): the product fraction N_C/N_A(0) early in the
     chemical relaxation, against the timestep, for the engine's fixed operator order and the randomized one
     (rx/<order>/short, tau_ladder_rx2.sh)
  b  the ternary relaxation of Fig. 2 (bath + 64 bodies of 20 nm): the bodies' mean squared displacement at the end of
     the run, against the timestep, for the fixed and the random order (ternary/{fixed,perm}, tau_ladder_ternary.sh)
(The bath's stripe-mode amplitude was a third panel until 2026-10-05; the owner kept the body MSD alone.)

Lines are least-squares fits linear in the timestep, weighted by the standard errors; their intercepts are each
scheme's tau -> 0 limit. The build prints every number the SI text quotes.
"""
import argparse, glob, json, os, shutil, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import jcp_style as J
import kip3_palette as P

DATA = os.path.join(HERE, '..', 'data')
PLOTS = os.path.join(HERE, '..', 'plots')
FIGD = os.environ.get('MANUSCRIPT_FIGURES', '')   # --install target: the manuscript's figures/ (set it to use --install)
RX_DT = (('8e-8', 0.08), ('4e-8', 0.04), ('2e-8', 0.02), ('1e-8', 0.01))       # file tag, us (80 to 10 ns)
T3_DT = (('50', 50.0), ('25', 25.0), ('12.5', 12.5), ('6.25', 6.25))            # file tag, us
# The two step orders the SI shows. The ladder also ran the smear refresh after the moves (ternary/smear,
# ternary/permsmear: no effect either); the SI does not show it, so the figure does not read it.
SCHEMES = (('fixed', 'fixed order', P.ARM['exc'], 'o'), ('perm', 'random order', P.ARM['sw'], 's'))
RX_FRAME = 1            # short window, frame 1 = tau 0.0004 (0.4 us): early in the chemical relaxation
PERIOD = 8


# Canvas and panel geometry, in inches from the top-left corner (FIGURE_STYLE.md section 5).
FW, FH = 7.2, 2.92
AX_Y, AX_W, AX_H = 0.42, 2.62, 2.00
AX_X = (0.98, 4.48)                    # left edges of panels a and b
LET_X = (0.05, 3.85)                   # their letters, left of each panel's y label


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


def rx_point(order, tag):
    """N_C/N_A(0) at RX_FRAME, per seed, for one order and timestep."""
    vals, tau = [], None
    for f in sorted(glob.glob(os.path.join(DATA, 'rx', order, 'short', 'rx_dt%s_s*.npz' % tag))):
        z = np.load(f, allow_pickle=True); c = z['counts'].astype(float); tau = float(z['tau'][RX_FRAME])
        vals.append(c[RX_FRAME, 2] / c[0, 0])
    return vals, tau


def body_msd(pos, dims):
    """Mean squared displacement (voxels^2) of the bodies from the first to the last frame, minimal-image increments."""
    L = np.array(dims, float)
    d = np.diff(pos, axis=0)
    d -= L * np.round(d / L)
    disp = d.sum(axis=0)
    return float((disp ** 2).sum(axis=1).mean())


def t3_point(scheme, tag):
    msds = []
    for f in sorted(glob.glob(os.path.join(DATA, 'ternary', scheme, 't3_tau%s_s*.npz' % tag))):
        z = np.load(f, allow_pickle=True); m = json.loads(str(z['meta']))
        msds.append(body_msd(z['posC'], m['dims']) * m['h_nm'] ** 2)
    return msds


def fit(x, y, e):
    p = np.polyfit(x, y, 1, w=1.0 / np.maximum(np.asarray(e), 1e-15))
    return p


def panel(ax, xs, rows, label, unit):
    """rows: list of (name, color, marker, means, sems). Points with error bars and their linear-in-dt fits."""
    xx = np.linspace(0, max(xs) * 1.05, 50)
    for name, col, mk, m, e in rows:
        ax.errorbar(xs, m, yerr=e, fmt=mk, color=col, mfc='white', mew=1.2, ms=5.5, capsize=2, elinewidth=1.0,
                    label=name)
        p = fit(xs, m, e)
        ax.plot(xx, np.polyval(p, xx), '--', color=col, lw=1.0)
        print('    %-20s %s   ->0 %.5f  slope %+.3e per %s' % (name, ' '.join('%.5f+-%.5f' % (a, b) for a, b in zip(m, e)),
                                                             p[1], p[0], unit))
    ax.set_xlim(0, max(xs) * 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(which='both', top=True, right=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(FW, FH)
        ax = [J.pub_ax(fig, x, AX_Y, AX_W, AX_H) for x in AX_X]

        print('a  reversible binding, N_C/N_A(0) at the early frame:')
        xs = [d for _, d in RX_DT]; rows = []
        for order, name, col, mk in SCHEMES[:2]:
            pts = [rx_point(order, t) for t, _ in RX_DT]
            st = [ms(v) for v, _ in pts]
            rows.append((name, col, mk, [s[0] for s in st], [s[1] for s in st]))
        tau_rx = rx_point('fixed', RX_DT[-1][0])[1]
        panel(ax[0], xs, rows, 'a', 'us')
        ax[0].set_xlabel('timestep $\\Delta t$ (\u00b5s)')
        ax[0].set_ylabel(r'$N_\mathrm{C}/N_\mathrm{A}(0)$')   # read at tau_rx; the caption says so

        print('b, c  ternary relaxation:')
        xs = [d for _, d in T3_DT]; rows_b = []
        for sch, name, col, mk in SCHEMES:
            pts = [t3_point(sch, t) for t, _ in T3_DT]
            if not all(len(v) > 1 for v in pts):
                print('  %s incomplete' % sch); continue
            sm = [ms(v) for v in pts]
            rows_b.append((name, col, mk, [x[0] for x in sm], [x[1] for x in sm]))
        print('  body MSD at the end (nm^2):'); panel(ax[1], xs, rows_b, 'b', 'us')
        ax[1].set_xlabel('timestep $\\Delta t$ (\u00b5s)')
        ax[1].set_ylabel('body MSD (nm\u00b2)')
        ax[0].legend(loc='upper right')
        for x, letter in zip(LET_X, 'ab'):
            J.pub_letter(fig, letter, x, 0.08)
        J.pub_check(fig, 'tau_ladder', si=True); J.check_text_overlap(fig, 'tau_ladder')
        for ext in ('pdf', 'png'):
            os.makedirs(os.path.join(PLOTS, ext), exist_ok=True)
            fig.savefig(os.path.join(PLOTS, ext, 'tau_ladder.' + ext), dpi=300)
        J.pub_fonts(os.path.join(PLOTS, 'pdf', 'tau_ladder.pdf'), 'tau_ladder')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'tau_ladder')
    print('wrote', os.path.normpath(os.path.join(PLOTS, '{pdf,png}', 'tau_ladder.*')))
    if A.install:
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTS, ext, 'tau_ladder.' + ext), os.path.join(FIGD, ext, 'tau_ladder.' + ext))
        print('installed into', FIGD)


if __name__ == '__main__':
    main()
