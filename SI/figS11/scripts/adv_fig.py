"""The supplementary advantage-metric figure: cost and advantage against the NUMBER OF REACTIONS.

Runs in the lm2.6 container, which has matplotlib. The x-axis is R, the number of independent
reversible pairs A_i + B_i <=> C_i of the pr* schemes (adv_schemes.py), all at the one packing
fraction ETA_PAIRS, the same 13 038 particles and the same sizes at every R; R = 0 is diffusion
only. The packing-fraction sweep this figure used to show is kept in adv_fig_eta_sweep.py.

Two lettered blocks, house style (studies/common/jcp_style.py), sans:

  A  the speed ratio against R, with the best single-core particle kernel in the numerator and each
     lattice arm in turn in the denominator (CPU lattice on one core, GPU lattice).
  B  the advantage metric A = speed ratio x ceiling ratio at one core with matched physics, linear.

The lattice ceiling tau_L = min(tau_samp, tau_rxn) enters B but is not drawn (owner, 2026-09-30);
the build prints it per point so the text can quote it.

usage (in the container):  python adv_fig.py   (after adv_analyze.py has written runs/summary.json)
"""
import json, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))   # vex-rddme/common
import jcp_style as J
import adv_schemes as S
import kip3_palette as P

# Publication style (FIGURE_STYLE.md section 5): Arial via PUB_RC, letters only, the canvas in inches.
W, H = 7.2, 2.80                 # in; two plots side by side
PX, PW, PY, PH = (0.72, 4.42), 2.68, 0.30, 2.05   # left edges, width, top, height of the two plots

RUNS = os.path.join(HERE, '..', 'data', 'runs')
OUT = os.path.join(HERE, '..', 'plots')


def load():
    """Rows of the reaction-count sweep, ordered by R."""
    rows = json.load(open(os.path.join(RUNS, 'summary.json')))
    by = {r['scheme']: r for r in rows if r['scheme'].startswith('pr')
          and abs(r['eta'] - S.ETA_PAIRS) < 1e-6}
    out = []
    for n in S.PAIR_COUNTS:
        r = by.get('pr%d' % n)
        if r is not None:
            out.append((n, r))
    return out


def main():
    pts = load()
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(W, H)
        axA, axB = (J.pub_ax(fig, x, PY, PW, PH) for x in PX)

        # ---- a: the speed ratio --------------------------------------------------------------
        # Numerator fixed at the best single-core particle kernel, so within the panel the only
        # thing that changes between the two curves is which lattice hardware runs the same work.
        # Colours and markers as the SI's other result figures: MPD-RDDME navy, its second arm teal.
        ARMS = [('core_best', 'CPU lattice (one core)', '-', 'o', P.ARM['exc'], 'white'),
                ('bestcore_vs_gpu', 'GPU lattice', '--', 's', P.ARM['sw'], None)]
        for key, lab, ls, mk, col, mfc in ARMS:
            xs = [n for n, r in pts if 'cost_ratio_' + key in r]
            ys = [r['cost_ratio_' + key] for n, r in pts if 'cost_ratio_' + key in r]
            if xs:
                axA.plot(xs, ys, ls, marker=mk, ms=5.5, color=col, mfc=mfc or col, mew=1.2, lw=1.5,
                         label=lab)
        axA.axhline(1.0, color=P.INK_DIM, lw=0.8)
        axA.set_yscale('log')      # the two arms differ by a factor of about twenty
        axA.set_ylabel(r'speed ratio $c_{\mathrm{R}}/c_{\mathrm{L}}$')
        axA.legend(loc='center right', handlelength=2.0)

        # ---- b: the advantage, one core, matched physics ------------------------------------
        xs = [n for n, r in pts if 'A_core_matched' in r]
        axB.plot(xs, [r['A_core_matched'] for n, r in pts if 'A_core_matched' in r], 'o-',
                 color=P.ARM['exc'], mfc='white', mew=1.2, ms=5.5, lw=1.5)
        axB.set_ylim(bottom=0)
        axB.set_ylabel(r'advantage $A$, one core')

        for ax, letter in ((axA, 'a'), (axB, 'b')):
            ax.set_xlabel(r'reversible bimolecular reactions $R$')
            ax.set_xticks(list(S.PAIR_COUNTS))
            ax.set_xlim(-0.3, max(S.PAIR_COUNTS) + 0.3)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(top=True, right=True, which='both')
        for x, letter in zip(PX, 'ab'):            # letters at each panel's top-left, one baseline
            J.pub_letter(fig, letter, x - 0.70, 0.02)

        os.makedirs(OUT, exist_ok=True)
        J.pub_check(fig, 'adv_metric', si=True)
        J.check_text_overlap(fig, 'adv_metric')
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(OUT, 'adv_metric.' + ext), dpi=300)
        J.pub_fonts(os.path.join(OUT, 'adv_metric.pdf'), 'adv_metric')
        import panel_export                            # each panel alone, no letter: ../plots/panels/
        panel_export.export(fig, 'adv_metric')
    print('wrote %s/adv_metric.{pdf,png}' % OUT)

    # The ceiling is no longer drawn, so the build prints what the text quotes.
    print('%3s %6s %9s %9s %9s %8s %8s %8s %9s' % ('R', 'n0', 'tsamp/us', 'trxn/us', 'tL/us',
                                                  'core_m', 'core_b', 'gpu_b', 'A_core'))
    for n, r in pts:
        print('%3d %6d %9.1f %9.3g %9.1f %8.3g %8.3g %8.3g %9.4g'
              % (n, r['n0'], r['tau_samp_s'] * 1e6, r['tau_rxn_s'] * 1e6, r['tau_L_s'] * 1e6,
                 r.get('cost_ratio_core_matched', float('nan')),
                 r.get('cost_ratio_core_best', float('nan')),
                 r.get('cost_ratio_bestcore_vs_gpu', float('nan')),
                 r.get('A_core_matched', float('nan'))))


if __name__ == '__main__':
    main()
