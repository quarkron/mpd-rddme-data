"""Equilibrium-constant shift with packing, scored against ReaDDy.

ReaDDy (Doi + detailed-balance handler, soft spheres at the Barker-Henderson diameter) is the
reference; the lattice arms come from the canonical-build sweep in rxK_out. The lattice shift is
the ratio of its exclusion run to its own ideal-gas run at identical rates, which cancels the
rate convention; the ideal control is plotted and sits on unity. Total packing is conserved by
the reaction because sigma_C^3 = sigma_A^3 + sigma_B^3.
"""
import glob, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))   # jcp_style
import jcp_style
import kip3_palette as PAL   # main-text arm colours: reference grey, MPD-RDDME navy

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')

# ReaDDy sweep at n0 matched to the lattice packing fractions, ten seeds per density;
# eta = 2 n0 (pi/6) sigma^3 / L^3, conserved by the reaction
rd = {}
NSEED = 10   # uniform replicate count: the a-priori fixed set seeds 0-9 at every density
for f in glob.glob(os.path.join(HERE, '..', 'data', 'readdy_ref', 'outm', 'rd*.npz')):
    try:
        m = json.loads(str(np.load(f, allow_pickle=True)['meta']))
        if int(m.get('seed', 99)) < NSEED:
            rd.setdefault(m['n0'], []).append(m['K_eff_over_K'])
    except Exception:
        pass
L3 = json.loads(str(np.load(glob.glob(os.path.join(
    HERE, '..', 'data', 'readdy_ref', 'outm', 'rd_n96_s0.npz'))[0], allow_pickle=True)['meta']))['L'] ** 3
re_ = np.array([2 * n * (math.pi / 6) * 27.0 / L3 for n in sorted(rd)])
rm = np.array([np.mean(rd[n]) for n in sorted(rd)])
rs = np.array([np.std(rd[n], ddof=1) / math.sqrt(len(rd[n])) for n in sorted(rd)])

lat = {'excl': {}, 'ideal': {}}
for f in glob.glob(os.path.join(HERE, '..', 'data', 'mpd_crddme', 'rxK_out', 'sw_*.npz')):
    m = json.loads(str(np.load(f, allow_pickle=True)['meta']))
    if int(m.get('seed', 99)) >= NSEED:
        continue
    lat['ideal' if m['no_exclusion'] else 'excl'].setdefault(
        round(m['eta'], 4), []).append(m['K_eff'] / m['K_id'])
le = np.array(sorted(lat['excl']))
lE = [np.array(lat['excl'][e]) for e in le]
lI = [np.array(lat['ideal'][e]) for e in le]
lm = np.array([a.mean() / b.mean() for a, b in zip(lE, lI)])
ls_ = np.array([abs(a.mean() / b.mean()) * math.sqrt(
    (a.std(ddof=1) / math.sqrt(len(a)) / a.mean()) ** 2 +
    (b.std(ddof=1) / math.sqrt(len(b)) / b.mean()) ** 2) for a, b in zip(lE, lI)])
im = np.array([b.mean() for b in lI])
is_ = np.array([b.std(ddof=1) / math.sqrt(len(b)) for b in lI])

for e, v, s in zip(le, lm, ls_):
    j = int(np.argmin(np.abs(re_ - e)))
    print('eta %.3f lattice %.3f+-%.3f | readdy %.3f+-%.3f (eta %.3f) | recovered %.0f%%'
          % (e, v, s, rm[j], rs[j], re_[j], 100 * (v - 1) / (rm[j] - 1)))

# Publication style (FIGURE_STYLE.md section 5): Arial through PUB_RC, no usetex; one panel, so no letter.
W, H = 4.36, 3.18
with plt.rc_context(jcp_style.PUB_RC):
    fig = jcp_style.si_page(W, H)
    ax = jcp_style.pub_ax(fig, 0.52, 0.08, 3.72, 2.62)
    ax.tick_params(top=True, right=True)
    ax.errorbar(re_, rm, yerr=rs, fmt='v-', color=PAL.ARM['ref'], lw=1.5, ms=5.5, elinewidth=1.0,
                capsize=2.5, label='iPRD', zorder=2)
    ax.errorbar(le, lm, yerr=ls_, fmt='o', color=PAL.ARM['exc'], mfc='white', mew=1.2, ms=5.5,
                elinewidth=1.0, capsize=2.2, lw=0, label='MPD-RDDME', zorder=3)
    # The point-particle control (im, is_: K_eff/K = 1.00 at every density) is no longer drawn
    # (owner, 2026-10-01): it is the denominator of the MPD-RDDME ratio, and the line at 1 marks it.
    ax.axhline(1.0, color=PAL.INK_DIM, lw=0.8, zorder=0)
    ax.set_xlabel(r'crowder density $\eta$')
    ax.set_ylabel(r'$K_{\mathrm{eff}}/K$')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc='upper left', handlelength=1.6)
    jcp_style.pub_check(fig, 'rx_K_main', si=True)
    jcp_style.check_text_overlap(fig, 'rx_K_main')
    for e in ('pdf', 'png'):
        os.makedirs(os.path.join(FIGD, e), exist_ok=True)
        fig.savefig(os.path.join(FIGD, e, 'rx_K_main.' + e), dpi=300)
    jcp_style.pub_fonts(os.path.join(FIGD, 'pdf', 'rx_K_main.pdf'), 'rx_K_main')
    import panel_export                            # one panel: no letters, nothing written
    panel_export.export(fig, 'rx_K_main')
    print('-> rx_K_main.{pdf,png}')
