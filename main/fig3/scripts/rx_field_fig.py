"""The manuscript figure for the reversible reaction in a resolved sinusoidal field.

    python rx_field_fig.py [--out out/fig_rx_field] [--install]

One figure in three lettered blocks, following the application figures of the same paper:

    A  the setup in one cartoon: the 10 nm subvolume and what it holds, the reaction and its
       volume-conserving product, and the static sinusoid across the 320 nm box.
    B  the steady occupancy of reactant and product, folded onto one period of the field.
    C  two periods of the lattice, flat, at the start, in mid-relaxation and at steady state: the reactant occupancy
       (relative to its instantaneous mean) beside the free-energy landscape U the operators read (relative to its
       mean), U computed from the same frame (rx_field_data.plane_landscape).
The divergence from the reference in time (block C until 2026-10-05) is the SI figure fig:rx-field-kl-time,
SI/figS16/scripts/rx_field_kl_fig.py, which imports ARMS from here.

Only the blocks are lettered. Each plot names its own axes, the species is named inside the frame
and the key names the three descriptions, so inner letters would add nothing to look up.

Runs in the lm2.6 container; TeX Live must be on PATH or the type will not match the other
figures. See NOTES.md for the runs this reads and for what was measured before block C was drawn
the way it is.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))   # vex-rddme/common

import jcp_style
import kip3_palette as P
import rx_field_data as D
import rx_field_voxels as V
from rx_field_scheme import draw_scheme

# 6.5 in is the 468 pt text width of the manuscript's single-column preprint layout, NOT
# jcp_style.FULL (6.69 in, the two-column reprint width). Drawn at 6.69 the figure is scaled by
# 0.97 on the page and every 8 pt label falls under the AIP floor.
TEXTW, TEXTH_DRAW = 6.5, 4.95

# Block A's cartoon reaches the top edge of the canvas, so its one-line conclusion has nowhere to
# sit the way B, C and D's do (each of those has an empty letter row above it already). The canvas
# therefore grows by NOTE_BAND at the TOP, and every rectangle below keeps its distance from the
# BOTTOM through R()/YB(): nothing on the page moves or changes size, a strip simply appears above
# block A, and A ends up with the same letter row every other block has.
NOTE_BAND = 0.0
TEXTH = TEXTH_DRAW + NOTE_BAND
YS = TEXTH_DRAW / TEXTH


def R(rect):
    """An axes rectangle written in the drawing's own height, mapped onto the taller canvas."""
    x, y, w, h = rect
    return [x, y * YS, w, h * YS]


def YB(y):
    """A figure-fraction y (matplotlib measures from the bottom), mapped the same way."""
    return y * YS

# tag -> (label, colour, linestyle, marker, which plots of B it belongs in). The reference is
# the zero of the divergence by construction, so it appears only in the profiles.
ARMS = (('ref', 'iPRD', P.ARM['ref'], '-', '', ('profile',)),
        ('exc', 'MPD-RDDME', P.ARM['exc'], '-', 'o', ('profile', 'kl')),
        ('pt', 'point particles', P.ARM['pt'], '--', 's', ('profile', 'kl')))

# Two periods of the field, one stack of voxels deep. The three frames are the two ends of the
# relaxation and a point halfway through it: the reaction is over by tau ~ 5e-3, so what the
# middle frame shows is the field response half built, not the chemistry.
SNAPS = (('late_excl_s0.npz', 0.0), ('early_excl_s0.npz', 0.053),
         ('late_excl_s0.npz', 15.0))
NX = 2 * D.NPER               # sixteen voxels
VMIN, VMAX = 1.0, 3.4         # occupancy per voxel spanned by the block C colour ramp


# 0.24 in right of the letter and 0.035 in down: the difference in cap height between 13 pt and
# 8 pt, which lands the sentence on the letter's baseline rather than on its top edge.
NOTE_LINE = 0.185 / TEXTH      # the letter's top to the top of the sentence on the line below it


def LETTER(fig, blk, c, y, note=None):
    """A lowercase block letter at the left margin, ALONE on its line, and the block's conclusion
    on the line beneath it: nothing shares a horizontal level with a letter."""
    fig.text(0.010, y, blk(c.lower()), fontsize=13, fontweight='bold', ha='left', va='top')
    if note:
        jcp_style.block_note(fig, 0.010, y - NOTE_LINE, note)


def has_tex():
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def IN(x, y, w, h):
    """An axes rectangle in inches (x, y from the bottom left) -> figure fractions."""
    return [x / TEXTW, y / TEXTH, w / TEXTW, h / TEXTH]


def block_b(fig):
    """Block B: steady occupancy folded onto one period of the field."""
    geom = dict(pa=IN(0.8125, 2.55, 2.5025, 0.95), pc=IN(3.90, 2.55, 2.5025, 0.95))
    ax = {k: fig.add_axes(R(v)) for k, v in geom.items()}
    for a in ax.values():
        a.grid(alpha=0.25, lw=0.5, which='both')

    for name, pkey in (('A', 'pa'), ('C', 'pc')):
        a = ax[pkey]
        ph, pm, ps = D.iprd_profile(name)
        phw, pmw, psw = np.r_[ph, 1.0], np.r_[pm, pm[0]], np.r_[ps, ps[0]]
        if name == 'A':
            assert abs(phw[np.argmin(pmw)] - 0.25) < 0.07, 'field-phase folding is misaligned'
        a.fill_between(phw, pmw - psw, pmw + psw, color='0.82', lw=0, zorder=1)
        a.plot(phw, pmw, '-', color=P.ARM['ref'], lw=1.2, label='iPRD', zorder=2)
        for tag, lbl, col, ls, mk, panels in ARMS:
            if 'profile' not in panels or tag == 'ref':
                continue
            x, ym, ys = D.lattice_profile('excl' if tag == 'exc' else 'ideal', name)
            xw = np.r_[x, 1.0]
            a.errorbar(xw, np.r_[ym, ym[0]], yerr=np.r_[ys, ys[0]], fmt=mk, ls=ls, color=col,
                       mfc='white' if tag == 'exc' else col, mew=1.0, ms=3.6, lw=0.9,
                       elinewidth=0.9, capsize=1.5, label=lbl, zorder=3)
        a.set_xlabel(r'$x/\lambda$', labelpad=1.5)
        a.set_xlim(0, 1)
        a.set_ylim(0.52, 1.40)                      # headroom below the trough for the key
        # Top left: every arm passes through the species mean at x = 0 and climbs only after the
        # trough at a quarter period, so this corner is the one none of them enters.
        a.text(0.03, 0.96, r'species $\mathsf{%s}$' % name, transform=a.transAxes, ha='left',
               va='top')

    ax['pa'].set_ylabel(r'$\langle n\rangle / \bar{n}$')
    return ax


def block_c(fig):
    """Two periods of the lattice, flat, at three points along the relaxation: the reactant occupancy relative to its
    instantaneous mean (left) beside the free-energy landscape U relative to its mean (right), one cell per subvolume
    along the field, each cell the mean of its plane of 1024 subvolumes (rx_field_data.plane_landscape)."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    # Diverging around the mean, white at the mean: a flat strip reads white, so the build-up of the field response
    # is the contrast. Occupancy teal (below) -> orange (above); U sand (low) -> navy (high), as in fig:schematic.
    cm_n = LinearSegmentedColormap.from_list('occ', ['#2bb3a3', '#ffffff', '#d97321'])
    cm_u = LinearSegmentedColormap.from_list('U', ['#f5dfa0', '#ffffff', '#1d3a7a'])
    rows = [D.plane_landscape(os.path.join(D.LAT, fn), tw) for fn, tw in SNAPS]
    rn = max(np.abs(nA[:NX] / nA.mean() - 1).max() for nA, _, _ in rows)
    ru = max(np.abs(U[:NX] - U.mean()).max() for _, U, _ in rows)
    NR = Normalize(1 - rn, 1 + rn); NU = Normalize(-ru, ru)
    cw = 2.5025 / NX                                                     # cell edge, in
    cols = ((0.8125, cm_n, NR, lambda nA, U: nA[:NX] / nA.mean()),
            (3.90, cm_u, NU, lambda nA, U: U[:NX] - U.mean()))
    ytop = 1.46
    for i, (nA, U, tau) in enumerate(rows):
        y = ytop - i * 0.40
        for x0, cm, nm, val in cols:
            a = fig.add_axes(IN(x0, y, NX * cw, cw))
            a.pcolormesh(np.arange(NX + 1), [0, 1], val(nA, U)[None, :], cmap=cm, norm=nm,
                         edgecolors='white', linewidth=0.6)
            a.set_xlim(0, NX); a.set_ylim(0, 1); a.axis('off')
            a.add_patch(__import__('matplotlib').patches.Rectangle((0, 0), NX, 1, fill=False,
                                                                     ec=P.INK_DIM, lw=0.6, clip_on=False))
        fig.text(0.74 / TEXTW, (y + cw / 2) / TEXTH, r'$\tau = %g$' % round(tau, 3), ha='right', va='center',
                 fontsize=8.5)
    for x0, lab in ((0.8125, r'occupancy $\langle n_\mathsf{A}\rangle/\bar n_\mathsf{A}$'),
                    (3.90, r'landscape $U-\bar U$ (kT)')):
        fig.text((x0 + NX * cw / 2) / TEXTW, (ytop + cw + 0.07) / TEXTH, lab, ha='center', va='bottom', fontsize=8.5)
    tn = np.floor(100 * rn / 5) * 5 / 100; tu = np.floor(2 * ru) / 2     # end ticks inside the ramp
    for x0, cm, nm, ticks in ((0.8125, cm_n, NR, [1 - tn, 1.0, 1 + tn]), (3.90, cm_u, NU, [-tu, 0.0, tu])):
        cax = fig.add_axes(IN(x0 + 0.55, 0.22, NX * cw - 1.1, 0.07))
        cb = fig.colorbar(ScalarMappable(nm, cm), cax=cax, orientation='horizontal', ticks=ticks)
        cb.ax.tick_params(labelsize=7.5, length=2)
        cb.outline.set_linewidth(0.5)
    for x0 in (0.8125, 3.90):   # under the last row of each column
        fig.text(x0 / TEXTW, (ytop - 2 * 0.40 - 0.13) / TEXTH, r'two periods of the field, along $x$ $\longrightarrow$',
                 ha='left', va='center', fontsize=7.5, color=P.INK_DIM)
    print('  block C: occupancy range +-%.3f of the mean, U range +-%.2f kT' % (rn, ru))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default=os.path.join(HERE, '..', 'plots', 'fig_rx_field'))
    p.add_argument('--install', action='store_true',
                   help='also copy into the manuscript figure directories')
    A = p.parse_args()

    D.field_axis_check()
    tex = has_tex()
    rc = dict(jcp_style.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True,
               'ytick.right': True})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if tex
              else {'mathtext.fontset': 'cm'})
    rc = jcp_style.sans(rc, tex)                  # text AND math; see jcp_style.sans

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context(rc):
        # Under usetex matplotlib ignores fontweight: the weight has to be in the TeX.
        blk = (lambda c: r'\textbf{%s}' % c) if tex else (lambda c: c)
        fig = plt.figure(figsize=(TEXTW, TEXTH))

        ax_a = fig.add_axes(IN(0.2275, 3.83, 6.1425, 1.12))
        draw_scheme(ax_a, tex=tex)
        # A's letter and note sit in the band above the cartoon; the other three have a letter
        # row of their own already.
        LETTER(fig, blk, 'A', 0.998)

        axb = block_b(fig)
        # B is the occupancy row, C the divergence row, D the configurations: the letter sits at
        # the top left of the block it names.
        for y, c, note in (
                (3.88, 'B', 'MPD-RDDME reproduces equilibrium distributions predicted by iPRD with high accuracy'),
                (2.12, 'C', 'Particles are driven toward locations of energy minima')):
            LETTER(fig, blk, c, y / TEXTH, note)
        # Built by hand rather than harvested from the axes: errorbar hands back a container
        # whose legend entry carries the cap and the bar, which reads as a fourth symbol.
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=col, ls=ls, marker=mk, ms=3.6,
                          mfc='white' if tag == 'exc' else col, mew=1.0, lw=1.2, label=lbl)
                   for tag, lbl, col, ls, mk, _ in ARMS]
        # The key goes INSIDE the reactant profile, lower right. It used to sit on the letter row,
        # which block B's headline now needs in full; the band between that row and the panels is
        # 0.15 in, too short for a line of key. Lower right is the one corner of that panel no arm
        # enters: the trough is at a quarter period, on the left.
        for k_ in ('pa', 'pc'):   # the same key in every plot; lower right is empty in both
            axb[k_].legend(handles=handles, loc='lower right', fontsize=7.2, frameon=False,
                           handlelength=1.8, labelspacing=0.35, borderpad=0.2, handletextpad=0.5)

        block_c(fig)

        jcp_style.check_width(fig, TEXTW, 'rx_field')
        jcp_style.check_text_overlap(fig, 'rx_field')

        os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
        for ext in ('pdf', 'png'):
            fig.savefig('%s.%s' % (A.out, ext), dpi=300)
        import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
        import panel_export
        panel_export.export(fig, 'rx_field')
    print('wrote %s.pdf/.png  TEX=%s' % (A.out, tex))

    # The composition shift is quoted in the text but no longer plotted, so the build prints it:
    # a number in the manuscript should be reproducible by running the thing that made the figure.
    CNT = D.counts_all()
    for tag in ('ref', 'exc', 'pt'):
        t, c = CNT[tag]
        m = t >= 3.0
        print('  late counts  %-4s  N_A/N_A(0) %.4f   N_C/N_A(0) %.4f'
              % (tag, c[m, 0].mean(), c[m, 2].mean()))

    if A.install:
        figd = os.environ.get('MANUSCRIPT_FIGURES', '')
        for ext in ('pdf', 'png'):
            dst = os.path.join(figd, ext, 'rx_field.' + ext)
            shutil.copyfile('%s.%s' % (A.out, ext), dst)
            print('installed', os.path.normpath(dst))


if __name__ == '__main__':
    main()
