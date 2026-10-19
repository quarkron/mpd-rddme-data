"""Supplementary figure: the cross-species occupancy correlation of the ternary relaxation.

    python rxthree_rho_fig.py

This is the panel that used to sit in the main-text ternary figure. It moved here because the main
figure's argument is that the lattice reproduces the reference's marginals and the body's mobility,
and the correlation is a second, independent statement -- that a volume-excluded lattice produces a
NEGATIVE cross-species correlation at all, which the point-particle limit is structurally incapable
of -- that deserves its own frame rather than a sixth panel competing for the reader's attention.

WHAT IT SHOWS
    rho(n_A, n_B) over all voxels against time. The slab start puts both species in the same half, so
    rho begins strongly positive; as the box fills, a voxel that is crowded with A has less room for B
    and the correlation goes negative. The reference crosses zero and settles near -0.37. MPD-RDDME
    crosses at the same time and settles near -0.51, the same overshoot of the anticorrelation the
    binary mixture shows. The point-particle arm decays toward zero and never produces the negative
    branch, because with no exclusion the two species do not compete for anything.

The pooling is rxthree_data.prepare(), shared with the main-text figure, so "the same run at the same
time" means the same thing in both.
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rxthree_data as RD
import jcp_style as J
import kip3_palette as P

OUT = 'rxthree_rho'


def main():
    D = RD.prepare()
    grid, L, arms = D['grid'], D['L'], D['arms']
    rb, rb_sem = D['rb'], D['rb_sem']
    xt = [0, 0.5, 1, 2, 4]; xtl = ['0', '0.5', '1', '2', '4']

    with plt.rc_context(J.PUB_RC):
        # One panel making one point, so no letter (FIGURE_STYLE.md section 5) and a single-column size.
        fig = J.si_page(3.65, 2.82)
        ax = J.pub_ax(fig, 0.62, 0.10, 2.90, 2.25)
        ax.axhline(0.0, color=P.INK_DIM, lw=0.8)
        ax.fill_between(grid, rb - rb_sem, rb + rb_sem, color=RD.REF, alpha=0.25, lw=0)
        ax.plot(grid, rb, 'o-', color=RD.REF, mfc='white', mew=1.0, ms=4.0, label='BD')
        for key, lbl, col, sty in arms:
            ax.plot(grid, L[key]['rho'], sty, color=col, mfc=col, mew=0, ms=4.0, label=lbl)
        ax.set_xscale('symlog', linthresh=grid[1], linscale=0.5)
        ax.set_xlim(-0.02, grid[-1] * 1.15)
        ax.set_xticks(xt); ax.set_xticklabels(xtl)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(which='both', top=True, right=True)
        ax.set_xlabel(r'time $\tau$')
        ax.set_ylabel(r'correlation $\rho(n_\mathrm{A}, n_\mathrm{B})$')
        ax.grid(alpha=0.25, lw=0.5)
        # Lower left: every arm starts near +0.8 and only the two exclusion arms go negative, after
        # tau ~ 0.7, so that corner holds no curve (upper right sat on the point-particle arm).
        ax.legend(loc='lower left')
        J.pub_check(fig, OUT, si=True); J.check_text_overlap(fig, OUT)
        for e in ('pdf', 'png'):
            os.makedirs(os.path.join(RD.FIGD, e), exist_ok=True)
            fig.savefig(os.path.join(RD.FIGD, e, OUT + '.' + e), dpi=300)
        J.pub_fonts(os.path.join(RD.FIGD, 'pdf', OUT + '.pdf'), OUT)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export                            # a single panel: nothing is written
        panel_export.export(fig, 'rxthree_rho')
    print('-> %s.{pdf,png}' % OUT)
    RD.report(D)
    i_eq = len(grid) - 1
    print('rho at equilibrium: BD %+.3f | %s'
          % (rb[i_eq], ' '.join('%s %+.3f' % (a[0], L[a[0]]['rho'][i_eq]) for a in arms)))
    for key, lbl, col, sty in arms:                     # where each arm crosses zero
        r = np.asarray(L[key]['rho'], float)
        s = np.flatnonzero((r[:-1] > 0) & (r[1:] <= 0))
        print('  %-14s zero crossing at tau %s' % (lbl, ('%.2f' % grid[s[0] + 1]) if len(s) else 'none'))
    s = np.flatnonzero((rb[:-1] > 0) & (rb[1:] <= 0))
    print('  %-14s zero crossing at tau %s' % ('BD', ('%.2f' % grid[s[0] + 1]) if len(s) else 'none'))


if __name__ == '__main__':
    main()
