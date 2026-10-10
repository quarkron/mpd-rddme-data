"""WM6 / Figure 4 from the 10-seed replicates.

Error bars are the standard error of the mean over 10 independent seeds -- measured, not modeled.
(The Ornstein-Uhlenbeck estimate used previously is vindicated: measured per-run scatter over
predicted spans 0.68 to 1.51 across the 13 configurations with mean 1.02 and no systematic bias.
Replicates are kept because the SEM is sqrt(10) tighter and assumes nothing.)
  panel (a) frequency response: mean amplitude and lag vs omega/Dk against the analytic NESS.
  panel (b) timestep response at fixed omega, two lattice spacings: fractional deviation of the
            mean from analytic vs eps_tau^(1). eps_tau carries no h, eps_samp = D tau/h^2 does,
            so if the collapse tracked eps_tau the two series would coincide.
"""
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator, FixedFormatter

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

try:
    subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
    USE_LATEX = True
except Exception:
    USE_LATEX = False

rc = {'font.family':'serif','font.size':14,'axes.labelsize':16,'legend.fontsize':11,
      'xtick.labelsize':12,'ytick.labelsize':12,'lines.linewidth':1.6,'lines.markersize':7,
      'axes.linewidth':1.1,'xtick.direction':'in','ytick.direction':'in',
      'xtick.top':True,'ytick.right':True,'xtick.minor.visible':True,'ytick.minor.visible':True,
      'xtick.major.size':6.0,'ytick.major.size':6.0,'xtick.minor.size':3.0,'ytick.minor.size':3.0,
      'savefig.bbox':'tight','savefig.pad_inches':0.05}
rc.update({'text.usetex':True,'font.serif':['Computer Modern Roman']} if USE_LATEX
          else {'mathtext.fontset':'cm'})

# Main-text palette (kip3_palette.ARM, 2026-10-01): amplitude navy, phase teal, analytic grey.
C_AMPL, C_PHASE, C_REF = '#123f7a', '#2bb3a3', '#5b6b7a'

with plt.rc_context(rc):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    rr = np.logspace(np.log10(r_f.min()/1.6), np.log10(r_f.max()*1.6), 300)
    ax.plot(rr, 1/np.sqrt(1+rr**2), '-', color=C_REF, lw=1.6,
            label=r'analytic $Dk/\sqrt{(Dk)^2+\omega^2}$')
    ax.errorbar(r_f, A_f/C_AMP, yerr=eA_f/C_AMP, fmt='o', color=C_AMPL, mfc='white', mew=1.6,
                ms=9, capsize=3, elinewidth=1.2, label=r'measured $u/c$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\omega / Dk$', fontsize=19)
    ax.set_ylabel(r'amplitude $u/c$', fontsize=17, color=C_AMPL)
    ax.tick_params(axis='y', colors=C_AMPL)
    ax.grid(True, which='major', alpha=0.30, lw=0.5); ax.grid(True, which='minor', alpha=0.10, lw=0.4)
    axr = ax.twinx()
    axr.plot(rr, np.degrees(np.arctan(rr)), '--', color=C_REF, lw=1.6,
             label=r'analytic $\arctan(\omega/Dk)$')
    axr.errorbar(r_f, L_f, yerr=eL_f, fmt='s', color=C_PHASE, mfc='white', mew=1.6, ms=9,
                 capsize=3, elinewidth=1.2, label=r'measured $\delta$')
    axr.set_ylabel(r'phase lag $\delta$ (deg)', fontsize=17, color=C_PHASE)
    axr.tick_params(axis='y', colors=C_PHASE); axr.set_ylim(0, 135); axr.minorticks_on()
    ax.set_ylim(top=3.2)        # headroom: the key sits in a band above every curve
    hs, hr = ax.get_legend_handles_labels(), axr.get_legend_handles_labels()
    ax.legend(hs[0]+hr[0], hs[1]+hr[1], loc='upper center', ncol=2, frameon=True, framealpha=0.95,
              edgecolor='0.6', fontsize=10.5, borderpad=0.5, columnspacing=1.0)

    bx.errorbar(et_t, dA_t, yerr=eA_t/Aan_t, fmt='o-', color=C_AMPL, mfc='white', mew=1.6, ms=9,
                capsize=3, elinewidth=1.2, label=r'$h=%g$: amplitude' % h_F)
    bx.errorbar(et_t, dL_t, yerr=eL_t/Lan_t, fmt='s-', color=C_PHASE, mfc='white', mew=1.6, ms=9,
                capsize=3, elinewidth=1.2, label=r'$h=%g$: phase lag' % h_F)
    bx.errorbar(et_4, dA_4, yerr=eA_4/Aan_4, fmt='o--', color=C_AMPL, mfc=C_AMPL, mew=1.2, ms=7,
                alpha=0.85, capsize=3, elinewidth=1.2, label=r'$h=%g$: amplitude' % h_4)
    bx.errorbar(et_4, dL_4, yerr=eL_4/Lan_4, fmt='s--', color=C_PHASE, mfc=C_PHASE, mew=1.2, ms=7,
                alpha=0.85, capsize=3, elinewidth=1.2, label=r'$h=%g$: phase lag' % h_4)
    # eps_samp of the h=2 series, printed above the data rather than beside it: a secondary
    # log axis will not hold a fixed locator here, and inline annotations collided with the curves.
    es_t = np.array([0.10, 0.20, 0.40, 0.50])
    tr = mtransforms.blended_transform_factory(bx.transData, bx.transAxes)
    bx.text(et_t[0] / 1.28, 0.955, r'$\varepsilon^{\mathrm{samp}}\!:$', transform=tr,
            fontsize=11.5, ha='right', va='center', color='0.25')
    for x, v in zip(et_t, es_t):
        bx.text(x, 0.955, '%g' % v, transform=tr, fontsize=11.5, ha='center', va='center',
                color='0.25')
    bx.text(et_t[-1] * 1.30, 0.955, r'($h\!=\!2$; $/4$ at $h\!=\!4$)', transform=tr,
            fontsize=10, ha='left', va='center', color='0.45')
    bx.set_xscale('log'); bx.set_yscale('log')
    bx.set_xlim(et_t.min()/1.9, et_t.max()*2.6)
    lo = min(dA_t.min(), dL_t.min(), dA_4.min(), dL_4.min())
    hi = max(dA_t.max(), dL_t.max(), dA_4.max(), dL_4.max())
    bx.set_ylim(lo/7.0, hi*3.2)
    bx.set_xlabel(r'$\varepsilon^{(1)}_{\tau}$ at $|x_v-c| = 2\sigma$', fontsize=19)
    bx.set_ylabel(r'fractional deviation from analytic', fontsize=15)
    bx.grid(True, which='major', alpha=0.30, lw=0.5); bx.grid(True, which='minor', alpha=0.10, lw=0.4)
    bx.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='0.6', fontsize=10.5,
              ncol=2, columnspacing=1.0, borderpad=0.5)
    ax.text(0.0, 1.015, '(a)', transform=ax.transAxes, fontsize=17, ha='left', va='bottom')
    bx.text(0.0, 1.015, '(b)', transform=bx.transAxes, fontsize=17, ha='left', va='bottom')
    plt.tight_layout()
    fig.savefig(FIGD/'pdf'/'wm6_omega_tau.pdf'); fig.savefig(FIGD/'png'/'wm6_omega_tau.png', dpi=300)
    import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import panel_export
    panel_export.export(fig, 'wm6_omega_tau')
    print('  -> wm6_omega_tau.{pdf,png}')
