#!/usr/bin/env python3
"""Chain size, chain mobility, and what a crowded polymer does to the crowders. Four lettered blocks.

Each block carries a title that states what it shows or what it concludes, and blocks are separated by a rule; the
plots inside a block share that one title and are told apart by their axes.

  a  The model. A chain on the FCC edge orbit: subunits of one voxel, bonds of sqrt(2) voxels, a bending energy
     kappa(1 - cos) at each joint, with the two depletants that surround it drawn at true size -- a bath of 3 nm
     spheres that fits between the beads and 20 nm crowder bodies that do not.
  b  Size. <R^2> against the number of bonds M at two stiffnesses, and against kappa at fixed length, each against
     the closed form for the phantom chain and against the self-avoiding chain sampled exactly by rejection. The gap
     between the two references is the excluded volume; the gap from the sampled reference to the engine is the
     engine's error.
  c  Mobility. The centre-of-mass diffusion coefficient and the slowest Rouse time, each normalised by its own dilute
     value so the two depletants share an axis, against Brownian dynamics and against the engine's own collisional
     closure -- the mean thinning factor <m> it applied, with nothing fitted.
  d  Demixing. The end state of a field-free box holding 20 nm crowders, a 3 nm bath and an assembling polymer, beside the same
     scene with the particles made point-like. Nothing pulls on anything in either: the only difference is whether the
     particles have a size. Each arm is drawn TWICE -- a cut-away box for context and a 160 nm slice for evidence.
     ★The box alone cannot settle it: it is a threshold, so a handful of isolated polymer cells in the point-particle
     arm render as solid blocks and read as domains. The slice carries the contrast (p-c)/(p+c) itself, both arms on
     one colour axis, where a mixture is pale and only real separation reaches the ends of the bar.

Data: fcc_si/statics_{len,kappa}.npz, fcc_si/transport/T_e*.npz and fcc_si/crowd11/C11_*, all on the SI move set
(fcc_si_campaign.sh, structure-ERA HEAD 3fa3ae8), with the readdy_bath / readdy_crowd arms linked in from fcc/ (the 20 nm
off-lattice arm reads the 2 s RCL_ runs); demix/D_CS_both020_si_s1 and D_CVS_point020_si_s1
(demix_campaign.sh, arm C: crowders + bath), with the chain-length spread from demix_lengths.py. Linear axes throughout.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.lines import Line2D
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
# vex-rddme layout: shared modules (jcp_style, kip3_palette, fcc_exact, chain_geometry, fixed_chain_harness) in vex-rddme/common
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import jcp_style                                                         # noqa: E402
import kip3_palette as KP                                                # noqa: E402
import fcc_exact as FX                                                   # noqa: E402
sys.path.insert(0, HERE)
from chain_dynamics_campaign import modes, autocorrelation, relaxation_time   # noqa: E402  (the estimators, one place)
import demix_lengths                                                     # noqa: E402  (block d: chain-length spread)

TEXTW = 6.5          # the 468 pt text width of the manuscript's single-column preprint layout


def style():
    """House rc: usetex when latex is on PATH, and the sans-serif variant the manuscript figures use.

    jcp_style.apply() alone leaves usetex off, which renders every label in DejaVu Serif mathtext --
    the silent fallback that makes this figure's type disagree with every other one in the paper.
    """
    import subprocess
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
        tex = True
    except Exception:
        tex = False
        print('  WARNING: no latex on PATH; type will not match the other figures')
    rc = dict(jcp_style.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in'})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if tex
              else {'mathtext.fontset': 'cm'})
    return jcp_style.sans(rc, tex), tex


# Blocks b and c read the SI-move-set re-runs (fcc_si_campaign.sh); the off-lattice ReaDDy arms are linked into fcc_si
# unchanged, since they do not depend on the lattice move set.
DATA = os.environ.get('FCC_DATA', os.path.join(HERE, '..', 'data', 'fcc_si'))
DEMIX = os.environ.get('DEMIX_DATA', os.path.join(HERE, '..', 'data', 'demix'))
SLAB_RUNS = os.path.join(HERE, '..', 'data', 'demix_noneq', 'slab')   # block d: the demixed (slab) start
OUT = os.environ.get('FIG_OUT', os.path.join(HERE, '..', 'plots'))
H_NM, SIG_C, SIG_M, SIG_B = 10.0, 20.0, 10.0, 3.0
C_ENG, C_SAW, C_BD, C_REF = KP.ARM['exc'], KP.ARM['pt'], KP.ARM['sw'], KP.ARM['ref']
# ★ONE SCHEME FOR THE SPECIES, used in the diagram and in the demixing maps alike: the polymer is
# green and a crowder is blue wherever either is DRAWN. (The line colours in blocks b and c encode
# something else entirely -- a stiffness, an arm of the comparison -- and are left alone.)
C_POLY, C_CROWD = KP.ARM['pt'], KP.ARM['exc']
C_BATH = KP.ARM['ref']                              # the 3 nm bath: the palette's neutral slate
INK = KP.INK
MC = {0.0: 1.000, 0.02: 0.987, 0.05: 0.952, 0.10: 0.987, 0.15: 0.894}    # fcc_chain_ao_mc.py, ratios only


def rows(name):
    d = np.load(os.path.join(DATA, name), allow_pickle=True)
    out = []
    for i in range(int(d['n_rows'])):
        out.append({k: float(np.asarray(d['%d_%s' % (i, k)])) for k in
                    ('n', 'kappa', 'box', 'r2_engine', 'r2_err', 'r2_exact_phantom', 'r2_exact_saw', 'r2_exact_saw_err')})
    return out


def panel_setup(ax):
    """The model as a diagram: a few voxels of the lattice, a chain on the edge orbit, and the two
    depletants at true size against them.

    ★A SCHEMATIC, NOT A SCENE. This used to be a 260 nm box holding the real thing -- two thousand
    bath particles and six crowders at true size -- which is an honest picture and an unreadable one
    at 1.5 in: the chain was a blue scribble in a grey haze. What a reader needs here is the rule of
    the model, so the lattice is six cells across and EVERY LABEL SITS OUTSIDE THE GRID, in the band
    above it or the one below, with a leader line to what it names. Nothing is written over the
    lattice, and no two labels share a lane.
    """
    NX, NY = 6, 5
    for i in range(NX + 1):
        ax.plot([i, i], [0, NY], '-', color=KP.CAGE, lw=0.5, alpha=0.9, zorder=0)
    for j in range(NY + 1):
        ax.plot([0, NX], [j, j], '-', color=KP.CAGE, lw=0.5, alpha=0.9, zorder=0)

    # The chain: one subunit per voxel, at voxel CENTRES, bonded across face diagonals (b = sqrt2 h).
    pts = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5], [3.5, 1.5], [4.5, 2.5]])
    ax.add_collection(LineCollection([[pts[i], pts[i + 1]] for i in range(len(pts) - 1)],
                                     colors=C_POLY, linewidths=2.0, zorder=3, capstyle='round'))
    ax.add_collection(EllipseCollection(np.full(len(pts), 1.0), np.full(len(pts), 1.0),
                                        np.zeros(len(pts)), units='x', offsets=pts,
                                        transOffset=ax.transData, facecolors=C_POLY,
                                        edgecolors='white', linewidths=0.7, zorder=4))
    j = pts[3]                                                   # the joint whose angle is drawn
    ang = [np.degrees(np.arctan2(*(q - j)[::-1])) for q in (pts[2], pts[4])]
    ax.add_patch(matplotlib.patches.Arc(j, 1.05, 1.05, theta1=min(ang), theta2=max(ang),
                                        color='white', lw=0.9, zorder=5))

    # The 20 nm crowder: two voxels across, so it cannot sit in one. Upper left, clear of the chain.
    ax.add_patch(plt.Circle((1.5, 3.5), 1.0, facecolor=C_CROWD, edgecolor='none', alpha=0.55, zorder=2))
    # The 3 nm bath: a dozen at true size against the voxel, in the corner the chain leaves empty.
    bath = np.array([[3.3, 3.2], [4.1, 3.9], [3.7, 4.5], [4.8, 3.3], [5.4, 4.2], [4.4, 4.8],
                     [5.6, 3.0], [3.2, 4.0], [5.1, 4.7], [4.0, 2.9]])
    ax.add_collection(EllipseCollection(np.full(len(bath), 0.3), np.full(len(bath), 0.3),
                                        np.zeros(len(bath)), units='x', offsets=bath,
                                        transOffset=ax.transData, facecolors=C_BATH,
                                        edgecolors='none', alpha=0.9, zorder=2))

    # ---- the label bands: two lanes only, one above the grid and one below, each entry with a
    # leader to what it names. ★Nothing is written over the lattice: the bending label used to sit
    # on top of a subunit, and dark navy type on a dark navy bead is simply not there.
    lead = dict(arrowstyle='-', color=INK, lw=0.6, shrinkA=1, shrinkB=1)
    ax.annotate('', xy=(1.0, -0.30), xytext=(0.0, -0.30), annotation_clip=False,
                arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=0.8, shrinkA=0, shrinkB=0,
                                mutation_scale=5))
    ax.text(0.5, -0.42, r'$h = 10$ nm', fontsize=7, color=INK, ha='center', va='top')
    ax.text(2.55, -1.05, r'$b = \sqrt{2}\,h$', fontsize=7, color=C_POLY, ha='center', va='top')
    ax.annotate('', xy=(1.15, 0.85), xytext=(2.45, -1.02), annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=C_POLY, lw=0.6, shrinkA=2, shrinkB=2))
    ax.text(NX + 0.2, -0.42, r'$\kappa\,(1-\cos\theta)$', fontsize=7, color=INK, ha='right',
            va='top')
    ax.annotate('', xy=(3.90, 1.12), xytext=(4.95, -0.42), annotation_clip=False, arrowprops=lead)
    ax.text(0.0, NY + 0.60, '20 nm crowder', fontsize=7, color=INK, ha='left', va='bottom')
    ax.annotate('', xy=(1.5, 4.4), xytext=(0.7, NY + 0.55), annotation_clip=False, arrowprops=lead)
    ax.text(NX, NY + 0.60, '3 nm bath', fontsize=7, color=INK, ha='right', va='bottom')
    ax.annotate('', xy=(4.4, 4.8), xytext=(5.2, NY + 0.55), annotation_clip=False, arrowprops=lead)
    ax.set_xlim(-0.2, NX + 0.2); ax.set_ylim(-1.75, NY + 1.15)
    ax.set_aspect('equal'); ax.axis('off')


def panel_length(ax):
    r = rows('statics_len.npz')
    for kap, col in ((1.0, C_ENG), (4.0, C_SAW)):
        s = sorted([x for x in r if abs(x['kappa'] - kap) < 1e-6], key=lambda x: x['n'])
        m = np.array([x['n'] - 1 for x in s])
        ax.plot(m, [x['r2_exact_phantom'] for x in s], '-', color=col, lw=1.2, zorder=2)
        ax.errorbar(m, [x['r2_exact_saw'] for x in s], yerr=[x['r2_exact_saw_err'] for x in s], fmt='s', color=col,
                    ms=4.2, mfc='white', mew=1.1, lw=0, elinewidth=0.8, capsize=1.6, zorder=3)
        ax.errorbar(m, [x['r2_engine'] for x in s], yerr=[x['r2_err'] for x in s], fmt='o', color=col, ms=4.6,
                    mew=1.1, lw=0, elinewidth=0.8, capsize=1.6, zorder=4)
        # In axes coordinates, not beside the last point: at the last point the label sat ON the
        # line it names.
        xy, va = ((0.86, 0.62), 'top') if kap == 4.0 else ((0.72, 0.22), 'bottom')
        ax.text(xy[0], xy[1], r'$\kappa = %g$' % kap, color=col, fontsize=7, va=va,
                ha='right', transform=ax.transAxes)
    ax.plot([], [], '-', color=INK, lw=1.2, label='closed form (phantom)')
    ax.plot([], [], 's', color=INK, ms=4.2, mfc='white', mew=1.1, lw=0, label='self-avoiding walk (MC)')
    ax.plot([], [], 'o', color=INK, ms=4.6, mew=1.1, lw=0, label='MPD-RDDME')
    ax.set_xlabel('bonds $M$'); ax.set_ylabel(r'$\langle R^{2}\rangle$  (voxel$^{2}$)')
    ax.set_xlim(0, 70); ax.set_ylim(0, 1330)
    ax.legend(loc='upper left', fontsize=7, handlelength=1.6, labelspacing=0.3, frameon=False)


def panel_kappa(ax):
    s = sorted(rows('statics_kappa.npz'), key=lambda x: x['kappa'])
    k = np.array([x['kappa'] for x in s])
    # Same encoding as the panel beside it -- source by marker, in one ink -- because the two now share one key.
    # Colouring the arms here would make olive mean "sampled" next to a panel where olive means kappa = 4.
    ax.plot(k, [x['r2_exact_phantom'] for x in s], '-', color=INK, lw=1.2)
    ax.errorbar(k, [x['r2_exact_saw'] for x in s], yerr=[x['r2_exact_saw_err'] for x in s], fmt='s', color=INK,
                ms=4.2, mfc='white', mew=1.1, lw=0, elinewidth=0.8, capsize=1.6)
    ax.errorbar(k, [x['r2_engine'] for x in s], yerr=[x['r2_err'] for x in s], fmt='o', color=INK, ms=4.6,
                mew=1.1, lw=0, elinewidth=0.8, capsize=1.6)
    m = s[0]['n'] - 1
    ax.axhline((m * np.sqrt(2)) ** 2, color=INK, lw=0.8, ls=':')
    ax.text(0.15, (m * np.sqrt(2)) ** 2 * 0.93, r'rod limit $(Mb)^{2}$', fontsize=7, color=INK, va='top')
    ax.set_xlabel(r'bending stiffness $\kappa$  ($k_{\mathrm{B}}T$)'); ax.set_ylabel(r'$\langle R^{2}\rangle$  (voxel$^{2}$)')
    ax.set_xlim(-0.3, 8.3); ax.set_ylim(0, (m * np.sqrt(2)) ** 2 * 1.1)


def _unwrap(p, box):
    """The Brownian-dynamics frames are stored wrapped; a centre of mass computed from them jumps a box length and
    inflates the diffusion coefficient by an order of magnitude. Unwrap frame to frame before anything else."""
    d = np.diff(p, axis=0); d -= box * np.round(d / box)
    return np.concatenate([p[:1], p[:1] + np.cumsum(d, axis=0)], axis=0)


def _chains_of(f):
    """(fraction, dt, array of shape (frames, chains, subunits, 3) in voxels) from any of the four sources."""
    d = np.load(f, allow_pickle=True)
    if 'frames' in d.files:                                              # Brownian dynamics: one chain, stored wrapped
        box = float(np.asarray(d['box']).ravel()[0])
        return round(float(d['eta']), 3), float(d['dt_frame']), _unwrap(np.asarray(d['frames'], float), box)[:, None]
    if 'beads_vox' not in d.files:                                       # a run that kept only centres of mass
        return None
    b = np.asarray(d['beads_vox'], float)                                # lattice: (frames, chains, subunits, 3)
    if 'eta' in d.files:
        eta = round(float(d['eta']), 3); dt = float(d['tau_s']) * float(d['bench_steps']) / (b.shape[0] - 1)
    else:
        # A dry run of the crowder driver still carries phi_core: the profile is set before the count is zeroed, so the
        # fraction has to come from the crowders that are actually there, or the dilute run lands on top of a real point.
        eta = 0.0 if int(np.asarray(d['n_crowd'])[-1]) == 0 else round(float(d['arg_phi_core']), 3)
        dt = float(np.median(np.diff(d['times'])))
    return eta, dt, b


def _ladder(patterns, what, lags=40):
    """D_cm from the short-lag centre-of-mass MSD, or tau_1 from the first Rouse mode. One estimator per column,
    applied identically to the lattice and to the Brownian reference, so the comparison is of physics not of method."""
    out = {}
    files = [g for pat in ([patterns] if isinstance(patterns, str) else patterns) for g in sorted(glob.glob(pat))]
    for f in files:
        got = _chains_of(f)
        if got is None:
            continue
        eta, dt, b = got
        vals = []
        for c in range(b.shape[1]):
            p = b[:, c]
            if what == 'd':
                com = p.mean(axis=1); L = np.arange(1, lags + 1)
                msd = np.array([((com[k:] - com[:-k]) ** 2).sum(-1).mean() for k in L]) * H_NM ** 2
                vals.append(np.polyfit(L * dt, msd, 1)[0] / 6.0)
            else:
                # Lag window: a quarter of the run, never below 0.25 s, never above half. A fixed 0.25 s cap was a
                # ceiling on tau_1 once the SI move set (no rigid rotation) put the crowded tau_1 above ~0.1 s.
                nlag = min(max(int(round(0.25 / dt)), p.shape[0] // 4), p.shape[0] // 2)
                acf = autocorrelation(modes(p, 3)[:, 1], nlag)
                t1 = relaxation_time(acf, dt)
                if np.isfinite(t1):
                    vals.append(t1)
        if vals:
            out.setdefault(eta, []).append(np.mean(vals))
    if 0.0 not in out:                                                   # without its own dilute run an arm has no scale
        return None
    e = np.array(sorted(out)); m = np.array([np.mean(out[x]) for x in e])
    s = np.array([np.std(out[x], ddof=1) / np.sqrt(len(out[x])) if len(out[x]) > 1 else 0.0 for x in e])
    r = m[0] / m if what == 't' else m / m[0]                            # tau rises with crowding; report its inverse
    return e, r, r * (s / m)


# The off-lattice crowder arm takes its dilute point from the dry Brownian runs: a run with no depletant is the same
# chain whichever depletant was going to be added, and re-running it would only add noise.
ARMS_DYN = [
    ('transport/T_e*_s*.npz',                                     'lattice, 3 nm bath',          KP.ARM['exc'], 'o-',  None),
    ('readdy_bath/RB_e*_s*.npz',                                  'off-lattice, 3 nm bath',      KP.ARM['sw'],  'd--', 'white'),
    ('crowd11/C11_phi*_s*.npz',                                   'lattice, 20 nm crowders',     KP.ARM['pt'],  's-',  None),
    (['readdy_crowd/RCL_e*_s*.npz', 'readdy_bath/RB_e0_s*.npz'],  'off-lattice, 20 nm crowders', KP.ARM['ref'], '^--', 'white'),   # 2 s runs: the 0.3 s RC_ set reads tau_1 low
]


def _dyn(ax, what):
    """One panel of the dynamics pair: the same four arms, the same estimator, normalised to each arm's dilute run."""
    for pat, lab, col, fmt, mfc in ARMS_DYN:
        pats = [os.path.join(DATA, q) for q in ([pat] if isinstance(pat, str) else pat)]
        r = _ladder(pats, what)
        if r is None:
            continue
        e, m, se = r
        ax.errorbar(e, m, yerr=se, fmt=fmt, color=col, ms=4.6, mfc=mfc or col, mew=1.1, lw=1.3, elinewidth=0.8,
                    capsize=1.6, label=lab)
    ax.set_xlabel(r'volume fraction  $\eta$')
    ax.set_xlim(-0.008, 0.163); ax.set_ylim(0, 1.12)


def panel_mobility(ax):
    _dyn(ax, 'd')
    ax.set_ylabel(r'$D_{\mathrm{cm}}$ / dilute value')
    # No key inside: five long labels in a panel this size sat on the very curves they name. It goes
    # under the block instead, where it crosses nothing (_draw).
    return ax.get_legend_handles_labels()


def panel_rouse(ax):
    """The slowest Rouse mode. The closure is symmetric on every move, so it should stretch the internal relaxation by
    the same factor it slows the centre of mass, with nothing fitted."""
    _dyn(ax, 't')
    ax.set_ylabel(r'$\tau_1(0)\,/\,\tau_1$')


def contrast(P, C):
    """(p - c)/(p + c) on fields normalised by their own means: +1 all polymer, -1 all crowder, 0 mixed.

    ★Normalising each field by its OWN mean is what lets four times, and two arms, share one colour
    axis. The polymer is still assembling while the box separates, so a map of raw counts would
    darken with time for a reason that has nothing to do with demixing.
    """
    p = P / max(P.mean(), 1e-12)
    c = C / max(C.mean(), 1e-12)
    return (p - c) / (p + c + 1e-12)


DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    'demix', [C_CROWD, '#f2f4f7', C_POLY])
# Block d reads the SI-move-set runs (Sep 28: full kink, end-bond rotation, edge translation, face hop; no crankshaft, no
# rigid rotation; demix-engine build). DEMIX_TAG='' reproduces the earlier-move-set render (the Sep 25 figure).
DEMIX_TAG = os.environ.get('DEMIX_TAG', '_si')
SNAP_RUN = 'D_CS_both020%s_s1' % DEMIX_TAG    # 20 nm crowders (phi 0.20) + 3 nm bath (eta 0.05), volume exclusion, snapshots kept
POINT_RUN = 'D_CVS_point020%s_s1' % DEMIX_TAG # the same three-component scene with every particle made point-like
ARMS_D = ((SNAP_RUN, 'finite-size'), (POINT_RUN, 'point particles'))
SNAP_K = (1, 2, 3, 4, 5)             # t = 2, 4, 6, 8, 10 s (t = 0 holds no chain: a solid square that also set the colour axis)
SLAB = 2                             # cells averaged through the depth of a slice: 40 nm. A 160 nm slab averaged across the
                                     # domain scale itself and washed the domains out
CELL = 2                             # voxels per cell: 20 nm, smoothed over 30 nm below
CROP = 2                             # cells dropped at each face: chains do not wrap the periodic boundary, so every arm,
                                     # the control included, shows a crowder-rich rim that is not demixing


def _field(f, nx, cell=CELL):
    """A per-voxel count field -> smoothed counts on `cell`-voxel blocks."""
    nc = nx // cell
    b = np.asarray(f, float).reshape(nx, nx, nx).reshape(nc, cell, nc, cell, nc, cell).sum(axis=(1, 3, 5))
    return ndi.gaussian_filter(b, 1.5), nc


def _blocks(idx, nx, cell=CELL):
    """A list of occupied voxel indices -> the same thing."""
    f = np.zeros(nx ** 3, float)
    np.add.at(f, np.asarray(idx, np.int64), 1.0)
    return _field(f, nx, cell)


def snap_slice(run, k, cut_axis=1):
    """The contrast field through the middle of the box at snapshot k, and that snapshot's time."""
    d = np.load(os.path.join(DEMIX, run + '.npz'), allow_pickle=True)
    nx = int(d['arg_nx'])
    P, nc = _blocks(d['snap_chain_%d' % k], nx)
    # ★The two arms carry their crowders differently and a figure that forgets it draws nonsense.
    # With volume exclusion a crowder is a structure complex, so it appears in the overlay as an
    # occupied voxel (snap_body). As a POINT particle it is a byte-lattice particle that never
    # enters the overlay at all, and the hook stores its count per voxel instead (snap_bath). Read
    # the wrong one and the point arm comes out with no crowders anywhere -- a solid polymer-green
    # square at every time, which is what the first draft of this panel showed.
    bath = 'snap_bath_%d' % k
    C, _ = (_field(d[bath], nx) if bath in d.files else _blocks(d['snap_body_%d' % k], nx))
    k0 = nc // 2 - SLAB // 2
    sl = [slice(None)] * 3
    sl[cut_axis] = slice(k0, k0 + SLAB)
    f = contrast(P[tuple(sl)].mean(axis=cut_axis), C[tuple(sl)].mean(axis=cut_axis))
    return f[CROP:-CROP, CROP:-CROP], float(np.asarray(d['snap_t_%d' % k]))


def panel_demix(grid, cax):
    """The same five times for both arms, and the control's whole history beside them.

    ★WHY SLICES AND NOT A BOX. A thresholded 3-D box shows only which cells passed a threshold, so a
    scattering of isolated polymer cells renders as solid blocks and reads as domains -- the eye
    cannot tell a blob one cell thick from one ten cells thick, at any viewing angle. The slice
    carries the contrast itself, all ten panels on one colour axis, so the two arms can be compared
    square by square. ★At t = 0 there is no chain anywhere -- the polymer is still free monomer --
    so t = 0 is a solid crowder-coloured square; it is left out, because it also pinned the shared colour axis at
    +-1 and pushed every later slice into the pale middle of the scale.
    """
    fields = [[snap_slice(run, k) for k in SNAP_K] for run, _ in ARMS_D]
    vmax = max(float(np.percentile(np.abs(f), 98)) for row in fields for f, _ in row)
    vmax = float(np.ceil(vmax * 20) / 20)
    print('  slice contrast: shared colour axis +-%.2f' % vmax)
    im = None
    for r, (row, (run, lab)) in enumerate(zip(fields, ARMS_D)):
        for c, (ax, (f, t)) in enumerate(zip(grid[r], row)):
            im = ax.imshow(f.T, origin='lower', cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6); sp.set_color(KP.CAGE)
            if r == 0:
                ax.set_title(r'$t = %g$ s' % t, pad=2.5, fontsize=8)
            if c == 0:
                ax.text(-0.08, 0.5, lab, transform=ax.transAxes, rotation=90, ha='right',
                        va='center', fontsize=7.5, color=INK)
    # Numbers, not words: x = (p - c)/(p + c) with p, c the polymer and crowder counts per cell each divided by its box
    # mean, so the local ratio p/c = (1 + x)/(1 - x) -- 3 at x = 0.5, 7 at 0.75 -- and 0 is the box-average mixture.
    ticks = [x for x in np.arange(-1.0, 1.0001, 0.25) if abs(x) <= vmax + 1e-9]
    cb = ax.figure.colorbar(im, cax=cax, orientation='vertical', ticks=ticks)
    cb.ax.set_yticklabels(['%+.2f' % x if x else '0' for x in ticks])
    cb.set_label(r'$(p-c)/(p+c)$', fontsize=7.5, labelpad=2)
    for x, lab in ((vmax, 'polymer-rich'), (-vmax, 'crowder-rich')):
        r = (1 + abs(x)) / (1 - abs(x))
        cb.ax.text(0.5, 1.035 if x > 0 else -0.035, '%s\n($p/c=%s$)' % (lab, ('%.0f' % r) if x > 0 else '1/%.0f' % r),
                   transform=cb.ax.transAxes, ha='center', va='bottom' if x > 0 else 'top', fontsize=6.5, color=INK)
    cb.ax.tick_params(labelsize=7, length=2)



# ---- the block furniture: a bold letter, the block's conclusion, and a rule ---------------------
# Uppercase letters and jcp_style.block_note, so a reader moving between this figure and the four in
# the main text meets the same object: bold letter, bold sentence, no numbers in it. A block that
# shows a MODEL rather than a measurement gets the letter alone -- see the same rule in the others.
LETTER_LINE = 0.175          # in: the letter's baseline to the top of the sentence under it


def block(fig, x, y, letter, title, bold):
    """A lowercase block letter, ALONE on its line, and the block's conclusion on the next one.

    ★Nothing shares a horizontal level with a letter. Beside it, one block's bullet sat level with
    the next block's sentence and the two read as a single line -- "a  Without crowding, ..." -- as
    though the sentence belonged to the block on the left.
    """
    fig.text(x, y, bold(letter.lower()), fontsize=12, fontweight='bold', ha='left', va='baseline',
             color=INK)
    if title:
        jcp_style.block_note(fig, x, y - LETTER_LINE / FIGH, title, size=8.5, color=INK)


def rule(fig, y, x0=0.012, x1=0.988):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=INK, lw=0.5, alpha=0.30))


def main():
    rc, tex = style()
    with plt.rc_context(rc):
        _draw()


FIGH = 4.78


def AX(x, ytop, w, h):
    """An axes rectangle in INCHES from the top-left. Stating the layout in inches is what makes the
    clearances readable: a panel needs about 0.35 in beneath it for tick labels and an axis label,
    and a block needs 0.16 in above it for its headline, whatever the canvas turns out to be."""
    return [x / TEXTW, 1.0 - (ytop + h) / FIGH, w / TEXTW, h / FIGH]


def YT(ytop):
    return 1.0 - ytop / FIGH


def _draw():
    # Under usetex matplotlib ignores fontweight: the weight has to be in the TeX.
    bold = ((lambda t: r'\textbf{%s}' % t) if matplotlib.rcParams.get('text.usetex')
            else (lambda t: t))
    fig = plt.figure(figsize=(TEXTW, FIGH))

    # ---- A: chain statics without crowders, the two panels across the width. 2026-10-09: the setup diagram (the old
    # block a) and the demixing block (the old d) are now fig:phase (main/fig5, fig_phase.py).
    panel_length(fig.add_axes(AX(0.66, 0.56, 2.46, 1.24)))
    panel_kappa(fig.add_axes(AX(3.96, 0.56, 2.46, 1.24)))
    block(fig, 0.010, YT(0.20), 'A', 'Without crowding, chain conformation statistics agree with theory', bold)

    # ---- B: the two dynamics panels, with their key on a line of its own beneath them.
    rule(fig, YT(2.22))
    ax_m = fig.add_axes(AX(0.66, 2.86, 2.46, 1.20))
    handles, labels = panel_mobility(ax_m)
    panel_rouse(fig.add_axes(AX(3.96, 2.86, 2.46, 1.20)))
    block(fig, 0.010, YT(2.36), 'B',
          'Suppression of polymer movements due to crowding can be simulated', bold)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, YT(4.42)), ncol=3,
               fontsize=7, frameon=False, handlelength=1.8, columnspacing=1.6, labelspacing=0.35)

    jcp_style.check_width(fig, TEXTW, 'polymer_r2')
    jcp_style.check_text_overlap(fig, 'polymer_r2')
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        q = os.path.join(OUT, 'polymer_r2.%s' % ext)
        fig.savefig(q, dpi=300); print('wrote', q)
    import os as _os, sys as _sys                  # each panel alone, no letter or headline: ../plots/panels/
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import panel_export
    panel_export.export(fig, 'polymer_r2')


if __name__ == '__main__':
    main()
