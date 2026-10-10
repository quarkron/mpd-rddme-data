"""SI figure: the single-core reference reproduces the GPU solver it is timed against.

    python cpu_gpu_validation_fig.py [--install]

Two panels, in the style of the SI's other result figures (rxthree_rho, rx_K_main):

  a  the equilibrium field response, n_A(x)/mean folded onto one period of the sinusoid, for the
     three reaction rates without exclusion and the fastest with it: GPU solver as markers, the
     single-core reference as lines. One run per arm and case (out/gpubr_*.npz, out/val_slope_*.txt).
  b  the zero-field K_eff/K_id at the two densities of the gate, with and without exclusion, mean
     and standard error over ten seeds per arm: GPU from mpd_crddme/rxK_out (seeds 0-9), CPU from
     cpu_gpu_K_seeds.sh (seeds 0-9, same steps and rates).

The build prints every number the SI text quotes. Runs in the lm2.6 container with TeX Live on PATH.
"""
import argparse, glob, json, math, os, shutil, subprocess, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))   # vex-rddme/common
import jcp_style as J
import kip3_palette as P

OUTD = os.path.join(HERE, '..', 'data', 'benchmark_out')   # inputs (gpubr_*, val_slope_*)
PLOTD = os.path.join(HERE, '..', 'plots')
KSEEDS = os.path.join(HERE, '..', 'data', 'cpu_gpu_val')
RXK = os.path.join(HERE, '..', 'data', 'mpd_crddme', 'rxK_out')
NPER = 8                          # sites per period: lambda = 80 nm at h = 10 nm
TEXTW = 6.5
# (label, GPU npz, CPU profile, colour). The CPU files were named by what was being probed, not by
# rate; the pairing is by rate, checked against each file's own K_eff on every build (pair_check).
CASES = [(r'$k_{\mathrm{off}}=2\times10^{2}$ s$^{-1}$', 'gpubr_co200.npz', 'val_slope_mildrx.txt', P.ARM['pt'], ':'),
         (r'$k_{\mathrm{off}}=2\times10^{4}$ s$^{-1}$', 'gpubr_co20000.npz', 'val_slope_co20000.txt', P.ARM['pt'], '--'),
         (r'$k_{\mathrm{off}}=5\times10^{5}$ s$^{-1}$', 'gpubr_co500000.npz', 'val_slope_ideal.txt', P.ARM['pt'], '-'),
         (r'exclusion, $5\times10^{5}$ s$^{-1}$', 'gpubr_excl_co500000.npz', 'val_slope_excl.txt', P.ARM['exc'], '-')]
# Main-text colours: green the point-particle arm (the three rates, told apart by line style), navy the
# volume-excluded arm. Markers are the GPU solver, lines the single-core reference.


def style():
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5); tex = True
    except Exception:
        tex = False
        print('  WARNING: no latex on PATH; type will not match the other figures')
    rc = dict(J.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if tex
              else {'mathtext.fontset': 'cm'})
    return rc


def fold(p):
    p = np.asarray(p, float)
    return (p / p.mean()).reshape(-1, NPER).mean(axis=0)


def cpu_profile(name):
    rows = [l.split() for l in open(os.path.join(OUTD, name)) if l[0].isdigit()]
    tail = [l for l in open(os.path.join(OUTD, name)) if 'K_eff' in l and 'totals' in l][0]
    return np.array([float(r[1]) for r in rows]), float(tail.split('K_eff')[-1])


def gpu_profile(name):
    z = np.load(os.path.join(OUTD, name), allow_pickle=True)
    return z['prof3'][0, 0, :], json.loads(str(z['meta']))['K_eff']


def kseeds_cpu(arm, n0):
    v = []
    for f in sorted(glob.glob(os.path.join(KSEEDS, 'K_%s_%d_s*.txt' % (arm, n0)))):
        tail = [l for l in open(f) if 'totals' in l]
        if tail:
            v.append(float(tail[0].split('K_eff')[-1]) / 200.0)
    return np.array(v)


def kseeds_gpu(eta, excl):
    v = []
    for f in glob.glob(os.path.join(RXK, 'sw_*.npz')):
        m = json.loads(str(np.load(f, allow_pickle=True)['meta']))
        if int(m['seed']) < 10 and abs(m['eta'] - eta) < 1e-3 and m['no_exclusion'] != excl:
            v.append(m['K_eff'] / m['K_id'])
    return np.array(v)


def ms(v):
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--install', action='store_true'); A = ap.parse_args()
    with plt.rc_context(style()):
        fig = plt.figure(figsize=(TEXTW, 2.55))
        gs = fig.add_gridspec(1, 2, wspace=0.30, width_ratios=[1.25, 1], top=0.90, bottom=0.18,
                              left=0.085, right=0.985)
        ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

        ph = np.arange(NPER + 1) / NPER
        print('field response, folded n_A/mean, peak-to-trough (GPU | CPU), K_eff (GPU | CPU):')
        for lab, g, c, col, ls in CASES:
            pg, kg = gpu_profile(g); pc, kc = cpu_profile(c)
            fg, fc = fold(pg), fold(pc)
            ax.plot(ph, np.r_[fc, fc[0]], ls, color=col, lw=1.1, label=lab)
            ax.plot(ph, np.r_[fg, fg[0]], 'o', color=col, mfc='white', mew=0.9, ms=3.4)
            dev = np.max(np.abs(fc - fg)) / np.ptp(fg)
            print('  %-34s %.4f | %.4f   K %.1f | %.1f   max |CPU-GPU| / amplitude %.3f'
                  % (c, np.ptp(fg), np.ptp(fc), kg, kc, dev))
        ax.plot([], [], 'o', color=P.INK_DIM, mfc='white', mew=0.9, ms=3.4, label='GPU solver (markers)')
        ax.plot([], [], '-', color=P.INK_DIM, lw=1.1, label='single core (lines)')
        ax.set_xlim(0, 1)
        # The key needs a band of its own above the curves, which peak at 1.37: two columns there.
        ax.set_ylim(0.62, 1.78)
        ax.set_xlabel(r'$x/\lambda$'); ax.set_ylabel(r'$\langle n_\mathsf{A}\rangle/\bar n_\mathsf{A}$')
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(frameon=False, fontsize=6.6, loc='upper left', handlelength=2.2, labelspacing=0.25,
                  ncol=2, columnspacing=1.0, borderaxespad=0.3)

        print('zero-field K_eff/K_id, mean +- SEM over seeds (GPU | CPU):')
        for n0, nbar, eta in ((26214, 0.8, 0.0226), (101580, 3.1, 0.0877)):
            for arm, excl, mk in (('excl', True, 'o'), ('ideal', False, 's')):
                g = kseeds_gpu(eta, excl); c = kseeds_cpu(arm, n0)
                gm, gs_ = ms(g)
                col = P.ARM['exc'] if excl else P.ARM['pt']          # main-text arm colours
                bx.errorbar(nbar - 0.06, gm, yerr=gs_, fmt=mk, color=col, mfc='white', mew=0.9,
                            ms=4.0, capsize=1.8, elinewidth=0.9)
                line = '  n %.1f %-5s GPU %.4f +- %.4f (%d)' % (nbar, arm, gm, gs_, len(g))
                if len(c) > 1:
                    cm, cs = ms(c)
                    bx.errorbar(nbar + 0.06, cm, yerr=cs, fmt=mk, color=col, ms=4.0,
                                capsize=1.8, elinewidth=0.9)
                    line += ' | CPU %.4f +- %.4f (%d)   diff %.1f combined SE' % (
                        cm, cs, len(c), (cm - gm) / math.hypot(gs_, cs))
                print(line)
        bx.plot([], [], 'o', color=P.INK_DIM, mfc='white', mew=0.9, ms=4.0, label='GPU solver')
        bx.plot([], [], 'o', color=P.INK_DIM, ms=4.0, label='single core')
        bx.text(1.95, 1.33, 'exclusion', color=P.ARM['exc'], fontsize=7, ha='center')
        bx.text(1.95, 1.025, 'point particles', color=P.ARM['pt'], fontsize=7, ha='center', va='bottom')
        bx.axhline(1.0, color=P.INK_DIM, lw=0.7)
        bx.set_xlim(0.3, 3.6); bx.set_xticks([0.8, 3.1])
        bx.set_xlabel(r'mean reactant occupancy per subvolume $\bar n$')
        bx.set_ylabel(r'$K_{\mathrm{eff}}/K$')
        bx.grid(alpha=0.25, lw=0.4)
        bx.legend(frameon=False, fontsize=6.8, loc='upper left', handlelength=1.2, labelspacing=0.25)

        for a, letter in ((ax, 'a'), (bx, 'b')):
            box = a.get_position()
            fig.text(box.x0 - 0.075, 0.985, r'\textbf{%s}' % letter, fontsize=11, va='top', ha='left')
        J.check_width(fig, TEXTW, 'cpu_gpu_validation')
        J.check_text_overlap(fig, 'cpu_gpu_validation')
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(PLOTD, 'cpu_gpu_validation.' + ext), dpi=300, pad_inches=0.08)   # usetex under-measures b's x label
        import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export
        panel_export.export(fig, 'cpu_gpu_validation')
    print('wrote %s/cpu_gpu_validation.{pdf,png}' % PLOTD)
    if A.install:
        figd = os.environ.get('MANUSCRIPT_FIGURES', '')
        for ext in ('pdf', 'png'):
            shutil.copyfile(os.path.join(PLOTD, 'cpu_gpu_validation.' + ext),
                            os.path.join(figd, ext, 'cpu_gpu_validation.' + ext))
        print('installed into', os.path.normpath(figd))


if __name__ == '__main__':
    main()
