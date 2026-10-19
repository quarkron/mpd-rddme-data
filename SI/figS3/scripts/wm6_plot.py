"""WM6 / Figure 4 from the 10-seed replicates.

Error bars are the standard error of the mean over 10 independent seeds -- measured, not modeled.
(The Ornstein-Uhlenbeck estimate used previously is vindicated: measured per-run scatter over
predicted spans 0.68 to 1.51 across the 13 configurations with mean 1.02 and no systematic bias.
Replicates are kept because the SEM is sqrt(10) tighter and assumes nothing.)
  panel (a) frequency response: mean amplitude and lag vs omega/Dk against the analytic NESS.
  panel (b) timestep response at fixed omega, two lattice spacings: fractional deviation of the
            mean from analytic vs eps_tau^(1). eps_tau carries no h, eps_samp = D tau/h^2 does,
            so if the collapse tracked eps_tau the two series would coincide.
2026-10-19: redrawn at final size in the publication style (FIGURE_STYLE.md section 5, common/jcp_style.py PUB_RC):
Arial, 10/11 pt, bold 15 pt letters; the analytic formulas left the keys and the eps_samp row left panel (b) (the
caption carries both); (a)'s key sits under the panel.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator, FixedFormatter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))
import jcp_style as J                               # noqa: E402

REP  = Path(__file__).resolve().parent.parent / 'data'   # vex-rddme: rep_h{2,4}.npz in <fig>/data
FIGD = Path(__file__).resolve().parent.parent / 'plots'
# cols: omega_over_Dk tau_s amp amp_analytic lag_deg lag_deg_analytic width eps_tau amp_sem lag_sem n
z2, z4 = (np.load(str(REP / f), allow_pickle=True) for f in ('rep_h2.npz', 'rep_h4.npz'))
h_F, h_4 = float(z2['meta'][1]), float(z4['meta'][1])
C_AMP, OM_FIXED = float(z2['meta'][5]), float(z2['meta'][7])

def srt(R, col):                      # aggregation keyed on strings, so re-sort numerically
    return R[np.argsort(R[:, col])] if R.size else R
F, T2, T4 = srt(z2['freq'], 0), srt(z2['tstep'], 1), srt(z4['tstep'], 1)

def unpack(R):
    return (R[:,0], R[:,1], R[:,2], R[:,3], R[:,4], R[:,5], R[:,7], R[:,8], R[:,9], R[:,10])
r_f, tau_f, A_f, Aan_f, L_f, Lan_f, et_f, eA_f, eL_f, n_f = unpack(F)
_,  tau_t, A_t, Aan_t, L_t, Lan_t, et_t, eA_t, eL_t, n_t = unpack(T2)
_,  tau_4, A_4, Aan_4, L_4, Lan_4, et_4, eA_4, eL_4, n_4 = unpack(T4)
dA_t, dL_t = np.abs(A_t/Aan_t - 1), np.abs(L_t/Lan_t - 1)
dA_4, dL_4 = np.abs(A_4/Aan_4 - 1), np.abs(L_4/Lan_4 - 1)

print('seeds per point: h=%g %s | h=%g %s' % (h_F, set(n_t.astype(int)), h_4, set(n_4.astype(int))))
print(' (a) om/Dk   A_mean+-SEM        dev%%    sig | lag_mean+-SEM      dev%%')
for i in range(len(r_f)):
    print('     %6.2f  %8.4f+-%.4f  %6.3f %6.1f | %7.3f+-%.4f  %6.3f'
          % (r_f[i], A_f[i], eA_f[i], abs(A_f[i]/Aan_f[i]-1)*100,
             abs(A_f[i]-Aan_f[i])/eA_f[i], L_f[i], eL_f[i], abs(L_f[i]/Lan_f[i]-1)*100))
for lbl, tt, dA, eA, Aan in (('h=2', tau_t, dA_t, eA_t, Aan_t), ('h=4', tau_4, dA_4, eA_4, Aan_4)):
    print(' (b) %s  tau(us) %s' % (lbl, ' '.join('%6.0f' % (v*1e6) for v in tt)))
    print('          dev%%    %s' % ' '.join('%6.3f' % (v*100) for v in dA))
    print('          sigma   %s' % ' '.join('%6.1f' % v for v in dA/(eA/Aan)))

# Main-text palette (kip3_palette.ARM, 2026-10-01): amplitude navy, phase teal, analytic grey.
C_AMPL, C_PHASE, C_REF = '#123f7a', '#2bb3a3', '#5b6b7a'
# eps_samp of the four h = 2 points (h = 4: a quarter of these). Drawn above the data until 2026-10-19; the caption
# carries them now.
ES_T = np.array([0.10, 0.20, 0.40, 0.50])
print(' (b) eps_samp at h=%g: %s; at h=%g: %s' % (h_F, ES_T, h_4, ES_T / 4))

W, H = 6.5, 3.6
XA, XB, AW, AH, YT = 0.66, 4.12, 2.06, 2.3, 0.36          # in: (a) carries a second y axis on its right

with plt.rc_context(J.PUB_RC):
    fig = J.si_page(W, H)
    ax = J.pub_ax(fig, XA, YT, AW, AH)
    bx = J.pub_ax(fig, XB, YT, 2.3, AH)

    rr = np.logspace(np.log10(r_f.min()/1.6), np.log10(r_f.max()*1.6), 300)
    ax.plot(rr, 1/np.sqrt(1+rr**2), '-', color=C_REF, lw=1.5, label='analytic amplitude')
    ax.errorbar(r_f, A_f/C_AMP, yerr=eA_f/C_AMP, fmt='o', color=C_AMPL, mfc='white', mew=1.2,
                capsize=2.5, elinewidth=1.0, label='measured amplitude')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(rr[0], rr[-1])
    ax.set_xlabel(r'drive frequency $\omega / Dk$')
    ax.set_ylabel(r'amplitude $u/c$', color=C_AMPL)
    ax.tick_params(axis='y', which='both', colors=C_AMPL)
    ax.grid(True, which='major', alpha=0.25, lw=0.5)
    axr = ax.twinx()
    axr.plot(rr, np.degrees(np.arctan(rr)), '--', color=C_REF, lw=1.5, label='analytic phase lag')
    axr.errorbar(r_f, L_f, yerr=eL_f, fmt='s', color=C_PHASE, mfc='white', mew=1.2,
                 capsize=2.5, elinewidth=1.0, label='measured phase lag')
    axr.set_ylabel(r'phase lag $\delta$ (deg)', color=C_PHASE)
    axr.tick_params(axis='y', which='both', colors=C_PHASE); axr.set_ylim(0, 100); axr.set_yticks([0, 30, 60, 90])
    ax.set_ylim(0.05, 1.6)
    # (a)'s key under its axis title: inside, its curves cross the whole frame
    hs, hr = ax.get_legend_handles_labels(), axr.get_legend_handles_labels()
    ax.legend(hs[0] + hr[0], hs[1] + hr[1], loc='upper left', bbox_to_anchor=(-0.30, -0.25), ncol=2,
              columnspacing=1.0, handlelength=1.6, handletextpad=0.4, borderaxespad=0.0)

    bx.errorbar(et_t, dA_t, yerr=eA_t/Aan_t, fmt='o-', color=C_AMPL, mfc='white', mew=1.2,
                capsize=2.5, elinewidth=1.0, label=r'$h = %g$, amplitude' % h_F)
    bx.errorbar(et_t, dL_t, yerr=eL_t/Lan_t, fmt='s-', color=C_PHASE, mfc='white', mew=1.2,
                capsize=2.5, elinewidth=1.0, label=r'$h = %g$, phase lag' % h_F)
    bx.errorbar(et_4, dA_4, yerr=eA_4/Aan_4, fmt='o--', color=C_AMPL, mfc=C_AMPL, mew=1.0, ms=4,
                alpha=0.85, capsize=2.5, elinewidth=1.0, label=r'$h = %g$, amplitude' % h_4)
    bx.errorbar(et_4, dL_4, yerr=eL_4/Lan_4, fmt='s--', color=C_PHASE, mfc=C_PHASE, mew=1.0, ms=4,
                alpha=0.85, capsize=2.5, elinewidth=1.0, label=r'$h = %g$, phase lag' % h_4)
    bx.set_xscale('log'); bx.set_yscale('log')
    bx.set_xlim(et_t.min()/1.5, et_t.max()*1.5)
    lo = min(dA_t.min(), dL_t.min(), dA_4.min(), dL_4.min())
    hi = max(dA_t.max(), dL_t.max(), dA_4.max(), dL_4.max())
    bx.set_ylim(lo/2.0, hi*12.0)            # headroom: the key sits above every curve
    bx.set_xticks([0.2, 0.4, 0.8])
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.set_xlabel(r'temporal flatness $\varepsilon^{(1)}_{\tau}$')
    bx.set_ylabel('deviation from analytic')
    bx.grid(True, which='major', alpha=0.25, lw=0.5)
    bx.legend(loc='upper left', handlelength=1.6, handletextpad=0.4)
    J.pub_letter(fig, 'a', 0.02, 0.04)
    J.pub_letter(fig, 'b', XB - 0.66, 0.04)
    pdf = FIGD / 'pdf' / 'wm6_omega_tau.pdf'
    fig.savefig(pdf); fig.savefig(FIGD/'png'/'wm6_omega_tau.png', dpi=300)
    J.pub_check(fig, 'wm6_omega_tau', si=True)
    J.check_text_overlap(fig, 'wm6_omega_tau')
    J.pub_fonts(str(pdf), 'wm6_omega_tau')
    import panel_export
    panel_export.export(fig, 'wm6_omega_tau')
    print('  -> wm6_omega_tau.{pdf,png} (%.2f x %.2f in)' % (W, H))
