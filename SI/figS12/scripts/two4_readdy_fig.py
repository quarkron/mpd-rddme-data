"""Equilibrium two-species statistics across size ratios, scored against ReaDDy.

The ReaDDy arm is computed from the equilibrated frames (tau >= 2) of the out_two ensembles
(16 seeds per ratio, harmonic repulsion at the Barker-Henderson diameters); the lattice arm is
the vex_out2 runs. Layout matches the original BD-referenced figure: (a), (b) marginals per
ratio, (c) Fano factors, (d) cross-species correlation.
"""
import glob, json, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))   # jcp_style
import jcp_style
import kip3_palette as PAL   # the main-text palette

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
RD = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), 'readdy_ref', 'out_two')   # was cwd-relative
LAT = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), 'mpd_crddme', 'vex_out2')
PATS = {1.0: 'rt_r1_s*.npz', 1.5: 'rt_r1p5_s*.npz', 2.0: 'rt_s*.npz', 2.5: 'rt_r2p5_s*.npz'}

bd = {}
for r, pat in PATS.items():
    for f in sorted(glob.glob(os.path.join(RD, pat))):
        z = np.load(f, allow_pickle=True)
        m = z['tau'] >= 2.0
        a = z['occA'][m].astype(float).ravel(); b = z['occB'][m].astype(float).ravel()
        meta = {'A': {'fano': a.var() / a.mean()}, 'B': {'fano': b.var() / b.mean()},
                'cross': {'pearson': float(np.corrcoef(a, b)[0, 1])}}
        PA = np.bincount(z['occA'][m].ravel()) / float(z['occA'][m].size)
        PB = np.bincount(z['occB'][m].ravel()) / float(z['occB'][m].size)
        bd.setdefault(r, []).append((meta, PA, PB))

lat = {}
for f in sorted(glob.glob(os.path.join(LAT, 'two_r*.npz'))):
    d = np.load(f, allow_pickle=True)
    m = json.loads(str(d['meta']))
    m['cross'] = {'pearson': m['pearson']}
    lat[m['ratio']] = (m, d['PA'], d['PB'])

RATIOS = sorted(set(bd) & set(lat))
if not RATIOS:
    raise SystemExit('no overlapping ratios')
print('ratios:', RATIOS, '| readdy seeds:', {r: len(bd[r]) for r in RATIOS})


def agg(r, path):
    v = [path(m) for m, _, _ in bd[r]]
    a = np.array(v, float)
    return a.mean(), (a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0)


def pmean(r, which):
    ps = [p for _, pa, pb in bd[r] for p in [pa if which == 'A' else pb]]
    n = max(len(p) for p in ps)
    return np.mean([np.pad(p, (0, n - len(p))) for p in ps], axis=0)


# Main-text palette (2026-10-01): the size ratio is an ordered quantity, so its four values take the
# sequential ramp of the main-text colour map; engines take the main-text arm colours, ReaDDy grey and
# MPD-RDDME navy, with the species told apart by marker.
CLR = [PAL.STOPS[1], PAL.STOPS[3], PAL.STOPS[4], PAL.STOPS[5]]
C_REF, C_LAT = PAL.ARM['ref'], PAL.ARM['exc']

# Publication style (FIGURE_STYLE.md section 5): Arial via PUB_RC, bold letters only, the canvas in inches.
# A shared key of the four size ratios across the top; panel b carries the engine key of a and b in its empty corner.
W, H = 7.25, 6.30
XL, XR, PW, PH = 0.80, 4.40, 2.75, 2.10       # left edges of the two columns, plot width and height
Y1, Y2 = 0.92, 3.72                           # tops of the two rows of plots
LDY = 0.27                                    # letter top above its plot's top
with plt.rc_context(jcp_style.PUB_RC):
    fig = jcp_style.si_page(W, H)
    aA, aB = jcp_style.pub_ax(fig, XL, Y1, PW, PH), jcp_style.pub_ax(fig, XR, Y1, PW, PH)
    aF, aR = jcp_style.pub_ax(fig, XL, Y2, PW, PH), jcp_style.pub_ax(fig, XR, Y2, PW, PH)
    for a_ in (aA, aB, aF, aR):
        a_.tick_params(top=True, right=True)

    for panel, which, lab in ((aA, 'A', r'$n_\mathrm{A}$'), (aB, 'B', r'$n_\mathrm{B}$')):
        for j, r in enumerate(RATIOS):
            P = pmean(r, which)
            k = 15 if which == 'A' else 8
            panel.plot(np.arange(min(k, len(P))), P[:k], 'o-', color=CLR[j], mfc='white',
                       mew=1.2, ms=4.8, lw=1.5, label='%g' % r)
            Q = lat[r][1] if which == 'A' else lat[r][2]
            panel.plot(np.arange(min(k, len(Q))), Q[:k], 's--', color=CLR[j], mfc=CLR[j],
                       mew=0, ms=4.2, lw=1.5, alpha=0.85)
        panel.set_xlabel('occupancy ' + lab)
        panel.set_ylabel(r'$P(%s)$' % lab.strip('$'))
        panel.grid(alpha=0.25, lw=0.5)
        panel.set_xlim(-0.4, 14 if which == 'A' else 7)
    aA.set_xticks(np.arange(0, 15, 3)); aB.set_xticks(np.arange(0, 8, 2))
    from matplotlib.lines import Line2D
    eng = [Line2D([], [], color=C_REF, marker='o', mfc='white', mew=1.2, ms=4.8, lw=1.5, ls='-', label='ReaDDy'),
           Line2D([], [], color=C_REF, marker='s', mew=0, ms=4.2, lw=1.5, ls='--', label='MPD-RDDME')]
    aB.legend(handles=eng, loc='upper right', handlelength=2.2)   # the empty corner of b; it reads for a too

    for which, mk, nm in (('A', 'o', 'A'), ('B', '^', 'B')):
        mu = np.array([agg(r, lambda m: m[which]['fano'])[0] for r in RATIOS])
        se = np.array([agg(r, lambda m: m[which]['fano'])[1] for r in RATIOS])
        aF.errorbar(RATIOS, mu, yerr=se, fmt=mk + '-', color=C_REF, mfc='white', mew=1.2, ms=5.0,
                    capsize=2.2, lw=1.5, label='ReaDDy (%s)' % nm)
        aF.plot(RATIOS, [lat[r][0][which]['fano'] for r in RATIOS], mk + '--', color=C_LAT,
                mfc=C_LAT, mew=0, ms=5.0, lw=1.5, label='MPD-RDDME (%s)' % nm)
    aF.set_xlabel(r'size ratio $\sigma_\mathrm{B}/\sigma_\mathrm{A}$')
    aF.set_ylabel('Fano factor')
    aF.grid(alpha=0.25, lw=0.5)
    # Room above the data for the key: the curves stay below 0.97, the key sits in the empty
    # upper-left corner where every curve is still near its x = 1 value of 0.63-0.67.
    aF.set_ylim(0.5, 1.18)
    aF.legend(ncol=1, loc='upper left', labelspacing=0.2, handlelength=2.0)

    mu = np.array([agg(r, lambda m: m['cross']['pearson'])[0] for r in RATIOS])
    se = np.array([agg(r, lambda m: m['cross']['pearson'])[1] for r in RATIOS])
    aR.errorbar(RATIOS, mu, yerr=se, fmt='o-', color=C_REF, mfc='white', mew=1.2, ms=5.0, capsize=2.2,
                lw=1.5, label='ReaDDy')
    aR.plot(RATIOS, [lat[r][0]['cross']['pearson'] for r in RATIOS], 's--', color=C_LAT,
            mfc=C_LAT, mew=0, ms=5.0, lw=1.5, label='MPD-RDDME')
    aR.axhline(0.0, color='k', lw=0.8, alpha=0.5)
    aR.set_xlabel(r'size ratio $\sigma_\mathrm{B}/\sigma_\mathrm{A}$')
    aR.set_ylabel(r'correlation $\rho(n_\mathrm{A},n_\mathrm{B})$')
    aR.grid(alpha=0.25, lw=0.5)
    aR.set_ylim(-0.95, 0.06)
    aR.legend(loc='upper right', bbox_to_anchor=(1.0, 0.91))   # the band between the zero line and the data

    h, l = aA.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5 * (XL + XR + PW) / W, 1.0 - 0.02 / H),
               ncol=len(RATIOS), title=r'size ratio $\sigma_\mathrm{B}/\sigma_\mathrm{A}$',
               title_fontsize=jcp_style.TEXT_PT, handlelength=2.2, columnspacing=1.6)
    for (x, y), t in zip(((XL, Y1), (XR, Y1), (XL, Y2), (XR, Y2)), 'abcd'):   # top-left, rows on one baseline
        jcp_style.pub_letter(fig, t, x - 0.78, y - LDY)
    jcp_style.pub_check(fig, 'two4_readdy', si=True)
    jcp_style.check_text_overlap(fig, 'two4_readdy')

    for e in ('pdf', 'png'):
        os.makedirs(os.path.join(FIGD, e), exist_ok=True)
        fig.savefig(os.path.join(FIGD, e, 'two4_readdy.' + e), dpi=300)
    jcp_style.pub_fonts(os.path.join(FIGD, 'pdf', 'two4_readdy.pdf'), 'two4_readdy')
    import panel_export                            # each panel alone, no letter: ../plots/panels/
    panel_export.export(fig, 'two4_readdy')
    print('-> two4_readdy.{pdf,png}')
    for r in RATIOS:
        fa = agg(r, lambda m: m['A']['fano']); fb = agg(r, lambda m: m['B']['fano'])
        pr = agg(r, lambda m: m['cross']['pearson'])
        print('r=%.1f readdy FanoA %.3f+-%.3f FanoB %.3f+-%.3f rho %.3f+-%.3f | lat %.3f %.3f %.3f'
              % (r, fa[0], fa[1], fb[0], fb[1], pr[0], pr[1],
                 lat[r][0]['A']['fano'], lat[r][0]['B']['fano'], lat[r][0]['cross']['pearson']))
