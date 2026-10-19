"""The model in four cartoons: the lattice in two views, the drive, and the binding kinetics.

    python kip3_scheme_fig.py --out out/fig_scheme

ONE PANEL, NOT FOUR. The figure is drawn in a single axes and carries no sub-labels, because it
is meant to sit as one panel inside a larger figure. Its three parts are laid out so that each
explains the next rather than standing alone:

    left    the axial view, looking down the tube: the 5x5 lattice, the 3x3 block the microtubule
            is inscribed in, and a motor's sidestep to the neighbouring lane. This is the view
            the tube renders cannot give, and it is what makes the lanes visible at all.
    right   the side view along the tube - the row of binding sites and a motor's step - with the
            potential that drives that step drawn DIRECTLY BENEATH IT, on the same x, so the
            staircase's steps line up with the binding sites they belong to. On the lattice the
            potential is a pure tilt, one step of -alpha per site with alpha = Delta_G_ATP / 2,
            so the hop is forward to four decimal places. The corrugation of the motor's own
            cycle is dashed because it is NOT in the lattice: that barrier sits below the grid
            and is subsumed into the hop rate, which is why k+ is set from the measured speed
            rather than from a barrier height.
    below   attachment and detachment, a birth-death process in a single voxel: a motor created
            at k_on per EMPTY site,
            removed at k_off on the lattice and at k_end at the plus end.

Runs in the lm2.6 container (numpy + matplotlib); jcp_style from studies/common.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M
import kip3_palette as P
from kip3_analyze import style
from kip3_snapshot3d import BODY_LO, BODY_HI, SHELL_LO, SHELL_HI, CENTRE, mix

RING = [(y, SHELL_LO) for y in range(SHELL_LO, SHELL_HI + 1)] + \
       [(SHELL_HI, z) for z in range(SHELL_LO + 1, SHELL_HI + 1)] + \
       [(y, SHELL_HI) for y in range(SHELL_HI - 1, SHELL_LO - 1, -1)] + \
       [(SHELL_LO, z) for z in range(SHELL_HI - 1, SHELL_LO, -1)]


def bead(ax, xy, r=0.42, z=4):
    from matplotlib.patches import Circle
    for k, (fr, off, tint) in enumerate([(1.00, 0.00, 0.00), (0.70, 0.22, 0.28),
                                         (0.38, 0.36, 0.58)]):
        ax.add_patch(Circle((xy[0] - off * r * 0.7, xy[1] + off * r * 0.7), fr * r,
                            facecolor=mix(P.MOTOR, '#ffffff', tint),
                            edgecolor=mix(P.MOTOR, '#000000', 0.5) if k == 0 else 'none',
                            lw=0.3 if k == 0 else 0.0, zorder=z + 0.01 * k))


AX_R, SV_H, TUBE_H = 1.5, 1.0, 1.7     # tube radius, site-row height, tube-band height
MOTOR_SITE = 2

# Two arrangements of the same cartoon. 'wide' stands alone; 'tall' is the portrait one that
# goes in the left column of the application figure, beside the results.
LAYOUT = dict(
    wide=dict(ax_c=(3.0, 7.5), sv_x0=8.5, sv_n=8, sv_y0=9.0, tube_y0=7.3, ph_top=6.4,
              ph_drop=0.42, kin_y=1.9, xlim=(-0.3, 16.9), ylim=(0.15, 11.0),
              side_label='right'),
    # In 'tall' the sidestep key goes BELOW the axial view rather than beside it: to its right
    # the cartoon is 10.7 voxels wide, which in a left-hand column pushes the key into the
    # neighbouring panel's y axis.
    # The axial view rides higher than the key needs so that the key, which hangs below it,
    # clears the side view's plus-end arrow.
    tall=dict(ax_c=(3.0, 14.4), sv_x0=0.5, sv_n=6, sv_y0=8.0, tube_y0=6.3, ph_top=5.3,
              ph_drop=0.45, kin_y=1.3, xlim=(-1.35, 6.7), ylim=(0.05, 17.9),
              side_label='below'),
    # 'pub' is the tall cartoon of the full-page figure (FIGURE_STYLE.md): words instead of rate
    # symbols and parameter values (those are in the caption), 10 pt type, and the axial view
    # moved down into the room the sidestep rate's label used to take.
    pub=dict(ax_c=(2.6, 13.75), sv_x0=0.5, sv_n=6, sv_y0=8.1, tube_y0=6.4, ph_top=4.75,
             ph_drop=0.45, kin_y=1.3, xlim=(-0.6, 8.2), ylim=(-0.65, 16.95),
             side_label='right', kin_x=(0.5, 5.0)),
)


def draw_scheme(ax, layout='wide'):
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
    L = LAYOUT[layout]
    pub = layout == 'pub'
    fs = 10 if pub else 7
    cx, cy = L['ax_c']
    SV_X0, SV_N, SV_Y0, TUBE_Y0 = L['sv_x0'], L['sv_n'], L['sv_y0'], L['tube_y0']
    PH_TOP, PH_DROP = L['ph_top'], L['ph_drop']
    edge = mix(P.CAGE, '#ffffff', 0.35)

    # ---- axial view -------------------------------------------------------------------------
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            ax.add_patch(Rectangle((cx + dy - 0.5, cy + dz - 0.5), 1, 1, facecolor=P.TUBE_FILL,
                                   edgecolor=edge, lw=0.4, zorder=1))
    for dy in (-2, -1, 0, 1, 2):
        for dz in (-2, -1, 0, 1, 2):
            if abs(dy) == 2 or abs(dz) == 2:
                ax.add_patch(Rectangle((cx + dy - 0.5, cy + dz - 0.5), 1, 1, facecolor='none',
                                       edgecolor=P.CAGE, lw=0.5, zorder=2))
    ax.add_patch(Circle((cx, cy), AX_R, facecolor=P.TUBE,
                        edgecolor=mix(P.TUBE, '#ffffff', 0.25), lw=0.5, zorder=3))
    ax.text(cx, cy, 'MT', fontsize=fs, ha='center', va='center', color='w', zorder=5)
    bead(ax, (cx + 2, cy))
    ax.add_patch(FancyArrowPatch((cx + 2.45, cy - 1.0), (cx + 2.45, cy + 1.0),
                                 connectionstyle='arc3,rad=0.62', arrowstyle='<|-|>',
                                 mutation_scale=8, lw=1.1, color=P.INK, zorder=6))
    # Under the arc's lower head, clear of the cell block: to the right of the arc the key would
    # need 2.4 voxels the tall layout's width does not have, and level with it, it sits on the
    # lattice.
    if pub:
        ax.text(cx + 3.3, cy, 'sidestep', fontsize=fs, ha='left', va='center', color=P.INK)
    else:
        ax.text(cx + 2.45, cy - 2.7, '$k_s$ =\n$0.3$ s$^{-1}$', fontsize=7, ha='center', va='top',
                color=P.INK, linespacing=1.35)
    ax.plot([cx - 2.5, cx - 1.5], [cy + 2.9] * 2, '-', color=P.INK, lw=0.8)
    for xv in (cx - 2.5, cx - 1.5):
        ax.plot([xv, xv], [cy + 2.75, cy + 3.05], '-', color=P.INK, lw=0.8)
    ax.text(cx - 1.35, cy + 2.9, '10 nm' if pub else r'$h$ = 10 nm', fontsize=fs, ha='left', va='center',
            color=P.INK)

    # ---- side view, with the potential that drives the step on the same x --------------------
    mx = SV_X0 + MOTOR_SITE + 0.5
    for xv in (mx - 0.5, mx + 0.5):
        ax.plot([xv, xv], [PH_TOP - 1.2, SV_Y0 + SV_H + 0.1], ':', color=P.INK_DIM, lw=0.5,
                alpha=0.8, zorder=1)
    ax.add_patch(Rectangle((SV_X0, TUBE_Y0), SV_N, TUBE_H, facecolor=P.TUBE,
                           edgecolor=mix(P.TUBE, '#ffffff', 0.25), lw=0.5, zorder=3))
    ax.text(SV_X0 + 0.5 * SV_N, TUBE_Y0 + 0.5 * TUBE_H, 'microtubule' if pub else 'microtubule, 30 nm',
            fontsize=fs,
            ha='center', va='center', color='w', zorder=5)
    for i in range(SV_N):
        ax.add_patch(Rectangle((SV_X0 + i, SV_Y0), 1, SV_H, facecolor='none', edgecolor=P.CAGE,
                               lw=0.5, zorder=2))
    bead(ax, (mx, SV_Y0 + 0.5 * SV_H))
    ax.add_patch(FancyArrowPatch((mx + 0.5, SV_Y0 + 0.5 * SV_H), (mx + 1.5, SV_Y0 + 0.5 * SV_H),
                                 arrowstyle='-|>', mutation_scale=8, lw=1.0, color=P.INK,
                                 zorder=6))
    ax.text(mx + 1.0, SV_Y0 + SV_H + 0.3, 'step' if pub else 'step, $k_+ = v_0/h$', fontsize=fs,
            ha='center', color=P.INK)
    PE_Y = SV_Y0 + SV_H + (1.35 if pub else 0.95)
    ax.annotate('', xy=(SV_X0 + SV_N - 0.2, PE_Y),
                xytext=(SV_X0 + SV_N - 1.9, PE_Y),
                arrowprops=dict(arrowstyle='-|>', lw=0.7, color=P.INK_DIM))
    ax.text(SV_X0 + SV_N - 2.1, PE_Y, 'plus end', fontsize=fs, ha='right',
            va='center', color=P.INK_DIM)

    lev = [PH_TOP - PH_DROP * i for i in range(SV_N)]
    for i in range(SV_N):
        ax.plot([SV_X0 + i, SV_X0 + i + 1], [lev[i]] * 2, '-', color=P.ARM['exc'], lw=1.5,
                zorder=4)
        if i < SV_N - 1:
            ax.plot([SV_X0 + i + 1] * 2, [lev[i], lev[i + 1]], '-', color=P.ARM['exc'], lw=1.5,
                    zorder=4)
    xs = np.linspace(SV_X0, SV_X0 + SV_N, 400)
    ax.plot(xs, PH_TOP - PH_DROP * (xs - SV_X0) + 0.11 * np.cos(2 * np.pi * (xs - SV_X0)), '--',
            color=P.INK_DIM, lw=0.7, alpha=0.85, zorder=3)
    ax.add_patch(Circle((mx, lev[MOTOR_SITE]), 0.30, facecolor=P.MOTOR,
                        edgecolor=mix(P.MOTOR, '#000000', 0.5), lw=0.4, zorder=7))
    ax.add_patch(FancyArrowPatch((mx + 0.36, lev[MOTOR_SITE] + 0.04),
                                 (mx + 0.95, lev[MOTOR_SITE + 1] + 0.12), arrowstyle='-|>',
                                 mutation_scale=8, lw=1.1, color=P.INK, zorder=8))
    ax.add_patch(FancyArrowPatch((mx - 0.36, lev[MOTOR_SITE] + 0.04),
                                 (mx - 0.95, lev[MOTOR_SITE - 1] + 0.12), arrowstyle='-|>',
                                 mutation_scale=5, lw=0.5, color=P.INK_DIM, zorder=8))
    if pub:
        ax.text(SV_X0 + 0.5 * SV_N, PH_TOP + 0.5, 'ratchet potential', fontsize=fs, ha='center',
                va='bottom', color=P.INK, zorder=9,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    else:
        ax.text(mx + 1.05, lev[MOTOR_SITE + 1] + 0.16, '$k_+$', fontsize=7, ha='left', va='bottom',
              color=P.INK)
        ax.text(mx - 0.95, lev[0] + 0.15, '$k_+e^{-\\alpha}$', fontsize=7, ha='center',
              va='bottom', color=P.INK_DIM)
        ax.annotate('', xy=(SV_X0 - 0.28, lev[1]), xytext=(SV_X0 - 0.28, lev[0]),
                  arrowprops=dict(arrowstyle='<|-|>', lw=0.7, color=P.INK, mutation_scale=6))
        ax.text(SV_X0 - 0.45, 0.5 * (lev[0] + lev[1]), r'$\alpha$', fontsize=8, ha='right',
              va='center', color=P.INK)
        ax.text(SV_X0 - 0.45, lev[0] + 0.25, r'$\Phi(s)$', fontsize=7, ha='right', va='bottom',
              color=P.INK)

    # ---- binding and unbinding: a birth-death process in one voxel ---------------------------
    y0 = L['kin_y']
    def site(x, occupied, dashed=False):
        ax.add_patch(Rectangle((x - 0.5, y0 - 0.5), 1, 1, facecolor='none', edgecolor=P.CAGE,
                               lw=0.6, ls='--' if dashed else '-', zorder=2))
        if occupied:
            bead(ax, (x, y0))
    xe, xm = L.get('kin_x', (1.0, 3.4))
    xc = 0.5 * (xe + xm)
    site(xe, False, dashed=True); site(xm, True)
    ax.add_patch(FancyArrowPatch((xe + 0.55, y0 + 0.3), (xm - 0.55, y0 + 0.3),
                                 connectionstyle='arc3,rad=-0.36', arrowstyle='-|>',
                                 mutation_scale=8, lw=1.0, color=P.INK, zorder=6))
    ax.add_patch(FancyArrowPatch((xm - 0.55, y0 - 0.3), (xe + 0.55, y0 - 0.3),
                                 connectionstyle='arc3,rad=-0.36', arrowstyle='-|>',
                                 mutation_scale=8, lw=1.0, color=P.INK, zorder=6))
    if pub:
        ax.text(xc, y0 + 1.0, 'attach', fontsize=fs, ha='center', va='bottom', color=P.INK)
        ax.text(xc, y0 - 1.0, 'detach', fontsize=fs, ha='center', va='top', color=P.INK)
        # (2026-10-19: the 'empty' / 'motor' words under the two boxes are gone; the caption names them)
    else:
        ax.text(2.2, y0 + 1.0, r'$k_{\mathrm{on}}$, per empty site', fontsize=7, ha='center',
                va='bottom', color=P.INK)
        ax.text(2.2, y0 - 1.0, r'$k_{\mathrm{off}}$  ($k_{\mathrm{end}}$ at the plus end)',
                fontsize=7, ha='center', va='top', color=P.INK)
        ax.text(1.0, y0 - 0.62, r'$\emptyset$', fontsize=7, ha='center', va='top', color=P.INK)
        ax.text(3.4, y0 - 0.62, 'M', fontsize=7, ha='center', va='top', color=P.INK)

    ax.set_xlim(*L['xlim']); ax.set_ylim(*L['ylim'])
    ax.set_aspect('equal'); ax.axis('off')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--layout', choices=tuple(LAYOUT), default='wide')
    p.add_argument('--out', required=True)
    A = p.parse_args()
    jcp = style()
    import matplotlib.pyplot as plt
    size = (jcp.FULL, 3.75) if A.layout == 'wide' else (jcp.COL, 5.0)
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    draw_scheme(ax, A.layout)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None)
    print('wrote %s.pdf/.png' % A.out)


if __name__ == '__main__':
    main()
