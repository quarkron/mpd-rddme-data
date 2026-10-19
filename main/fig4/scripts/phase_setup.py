"""fig:phase a -- the model of the phase-separation runs as one diagram, half the figure width (2026-10-09; replaces
the fig:polymer setup square that sat here first).

On a 10 x 5 patch of lattice: a 20 nm crowder and the 3 nm bath at true size against the 10 nm voxel; a chain
on one fcc sublattice (one subunit per voxel, bonds across face diagonals, bending at a joint), with a free monomer at
each tip and the growth/shrink pair k+ / k- drawn there; free monomers, and one nucleating into a single-subunit chain
at k_nuc. Each species carries its bare diffusion coefficient. The numbers are the run's (main.tex, sec. on demixing):
D monomer 0.3, crowder 0.04, bath 0.15 um^2/s.

As in fig_polymer_r2.panel_setup, the names sit in a band above the grid and the geometry in a band below, each with a
leader to what it names; only the rate labels sit on the lattice, on a white ground beside their arrows.
"""
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection, LineCollection

import fig_polymer_r2 as F

NX, NY = 10, 5
FS = 10            # pt: every label (FIGURE_STYLE.md). 2026-10-19: the D values, b = sqrt2 h and the bending energy
                   # kappa(1 - cos theta) left the panel for the caption; names, the 10 nm bar and the rates stay.


def _beads(ax, pts, face, edge, d=0.9, lw=0.8, z=4):
    pts = np.atleast_2d(np.asarray(pts, float))
    ax.add_collection(EllipseCollection(np.full(len(pts), d), np.full(len(pts), d), np.zeros(len(pts)), units='x',
                                        offsets=pts, transOffset=ax.transData, facecolors=face, edgecolors=edge,
                                        linewidths=lw, zorder=z))


def _pair(ax, a, b, top, bottom, off=0.13, lab=0.42):
    """Forward/backward harpoons from a to b, `top` written on the forward side and `bottom` on the backward one."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; u = d / np.linalg.norm(d); n = np.array([-u[1], u[0]])
    for s, (p, q) in ((1, (a, b)), (-1, (b, a))):
        ax.annotate('', xy=q + s * off * n, xytext=p + s * off * n,
                    arrowprops=dict(arrowstyle='-|>', color=F.INK, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0),
                    zorder=6)
    m = 0.5 * (a + b)
    box = dict(boxstyle='round,pad=0.08', fc='white', ec='none')
    ax.text(*(m + lab * n), top, fontsize=FS, color=F.INK, ha='center', va='center', bbox=box, zorder=7)
    ax.text(*(m - lab * n), bottom, fontsize=FS, color=F.INK, ha='center', va='center', bbox=box, zorder=7)


def _label(ax, x, y, text, target, ha='center', color=None, r=0.0, sx=None):
    """A name in a band outside the grid and a leader to what it names. The leader starts under (or over) the text at
    sx (default x) and STOPS AT THE EDGE of the named particle: target is its centre and r its radius."""
    ax.text(x, y, text, fontsize=FS, color=color or F.INK, ha=ha, va='bottom' if y > NY else 'top', multialignment=ha,
            linespacing=1.15)
    p0 = np.array([x if sx is None else sx, y + (-0.08 if y > NY else 0.08)])
    c = np.asarray(target, float)
    d = p0 - c
    end = c + r * d / np.linalg.norm(d)
    ax.annotate('', xy=tuple(end), xytext=tuple(p0), annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=color or F.INK, lw=0.6, shrinkA=0, shrinkB=0))


def panel_setup(ax):
    """10 x 5 cells (2026-10-09, compact: half the width). Crowder and bath top left, the chain along the bottom with a
    monomer at each tip, free monomers and nucleation at the right."""
    for i in range(NX + 1):
        ax.plot([i, i], [0, NY], '-', color=F.KP.CAGE, lw=0.5, alpha=0.9, zorder=0)
    for j in range(NY + 1):
        ax.plot([0, NX], [j, j], '-', color=F.KP.CAGE, lw=0.5, alpha=0.9, zorder=0)
    POLY, MONO = F.C_POLY, 'white'

    # ---- crowder and bath, top left
    ax.add_patch(plt.Circle((1.0, 4.0), 1.0, facecolor=F.C_CROWD, edgecolor='none', alpha=0.55, zorder=2))
    bath = np.array([[2.6, 3.5], [3.3, 4.2], [2.9, 4.7], [3.9, 3.6], [4.4, 4.5], [3.6, 3.1], [4.6, 3.9], [2.4, 4.3]])
    _beads(ax, bath, F.C_BATH, 'none', d=0.3, z=2)

    # ---- the chain along the bottom, one subunit per voxel centre, bonded across face diagonals
    ch = np.array([[2.5, 0.5], [3.5, 1.5], [4.5, 0.5], [5.5, 1.5], [6.5, 2.5]])
    ax.add_collection(LineCollection([[ch[i], ch[i + 1]] for i in range(len(ch) - 1)], colors=POLY, linewidths=2.0,
                                     zorder=3, capstyle='round'))
    _beads(ax, ch, POLY, 'white')
    j = ch[1]                                                    # the turn at a joint: incoming bond extended (dotted)
    ax.plot([j[0], j[0] + 0.75], [j[1], j[1] + 0.75], ':', color=F.INK, lw=0.8, zorder=5)
    ax.add_patch(mpatches.Arc(j, 1.0, 1.0, theta1=-45, theta2=45, color=F.INK, lw=0.8, zorder=5))
    ax.text(j[0] + 0.80, j[1] + 0.40, r'$\theta$', fontsize=FS, color=F.INK, ha='left', va='center', zorder=5)
    mL, mR = np.array([0.5, 1.5]), np.array([7.9, 4.3])        # a free monomer at each tip, and the growth pair there
    _beads(ax, [mL, mR], MONO, POLY, lw=1.1)
    _pair(ax, mL + [0.42, -0.30], ch[0] - [0.42, -0.30], r'$k_+$', r'$k_-$', lab=0.48)
    _pair(ax, mR - [0.33, 0.42], ch[-1] + [0.33, 0.42], r'$k_+$', r'$k_-$', lab=0.50)

    # ---- free monomers and nucleation, right
    _beads(ax, np.array([[9.4, 4.4], [8.8, 2.9]]), MONO, POLY, lw=1.1)
    a, b = np.array([7.6, 0.8]), np.array([9.4, 0.8])
    _beads(ax, a, MONO, POLY, lw=1.1); _beads(ax, b, POLY, 'white')
    ax.annotate('', xy=b - [0.5, 0], xytext=a + [0.5, 0],
                arrowprops=dict(arrowstyle='-|>', color=F.INK, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(0.5 * (a[0] + b[0]), a[1] + 0.32, r'$k_{\mathrm{nuc}}$', fontsize=FS, color=F.INK, ha='center',
            va='bottom', bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none'), zorder=7)

    # ---- names above, geometry below
    top, bot = NY + 0.40, -0.40
    _label(ax, 0.0, top, '20 nm crowder', (1.0, 4.0), ha='left', r=1.0, sx=0.9)
    _label(ax, 4.45, top, '3 nm bath', (4.4, 4.5), r=0.15)
    _label(ax, 6.6, top, 'chain\nsubunit', tuple(ch[-1]), r=0.45, sx=6.55)
    _label(ax, NX, top, 'free monomer', (9.4, 4.4), ha='right', r=0.45, sx=9.4)
    ax.annotate('', xy=(1.0, bot + 0.12), xytext=(0.0, bot + 0.12), annotation_clip=False,
                arrowprops=dict(arrowstyle='<|-|>', color=F.INK, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(0.5, bot, '10 nm', fontsize=FS, color=F.INK, ha='center', va='top')
    _label(ax, 3.0, bot, 'bond', (3.0, 1.0), color=POLY)
    _label(ax, 5.6, bot, 'bending', (3.97, 1.33))
    ax.text(8.5, bot, 'nucleation', fontsize=FS, color=F.INK, ha='center', va='top')
    ax.set_xlim(-0.2, NX + 0.2); ax.set_ylim(-1.05, NY + 1.05)
    ax.set_aspect('equal'); ax.axis('off')
