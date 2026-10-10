"""Two-row version of the Isaacson-Zhang figure-8 panel: complex profiles across the contact
zone at a sequence of times, point particles above and volume excluded below.

    python synapse_maps_fig.py --point out/M_react_point.npz --excluded out/M_react_excluded.npz \
        --out out/fig_maps_react --title 'reactive'

WHY SURFACE DENSITY AND NOT COUNTS
    Their colour axis is molecules per mesh polygon, and their polygon is 7093 nm^2 against our
    1600 nm^2 voxel, so the same physical density differs by 4.4x in raw counts between the two
    papers. Everything here is plotted as molecules per um^2, which is comparable to theirs after
    dividing their scale by the polygon area: their 0-20 per polygon is 0-2820 per um^2.

WHY BOTH ROWS SHARE ONE COLOUR AXIS
    The comparison is the point of the figure. Giving each row its own scale would hide exactly
    the effect being shown, which is that one description reaches densities the other cannot.
"""
import argparse, json, math, os, sys
import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d['meta']))
    if 'maps' not in d:
        raise SystemExit('%s has no maps; re-run with --map-times' % path)
    return d, meta


def coarsen(maps, k):
    """Block-sum the map by k in each direction. Our voxel is 40 nm and their mesh polygon is
    7093 nm^2, about 95 nm across, so displaying voxel-resolved counts shows Poisson speckle
    their figure does not have: at their density a 40 nm voxel holds about 2 complexes, so the
    shot noise is 70% per voxel. k = 2 gives 80 nm cells of 6400 nm^2, within 10% of their
    polygon area, which is the honest way to put the two figures side by side. The DENSITY is
    unchanged by this; only the sampling area is."""
    if k <= 1:
        return maps
    n = maps.shape[-1] // k * k
    m = maps[:, :n, :n]
    return m.reshape(m.shape[0], n // k, k, n // k, k).sum(axis=(2, 4))


def to_density(maps, meta, k=1):
    """Counts per cell -> molecules per um^2."""
    return maps / (meta['h_nm'] * k * 1e-3) ** 2


def masked(dens, meta, k=1):
    """Blank everything outside the disk so the panel is a disk, as in their figure."""
    n = dens.shape[-1]
    c = 0.5 * (n - 1)
    ix = np.arange(n)
    r = np.hypot(ix[:, None] - c, ix[None, :] - c) * (meta['h_nm'] * k * 1e-3)
    return np.where(r <= meta['r_disk'], dens, np.nan), r


RHO_JAM = 0.82 / (math.pi * (0.5 * 8.828e-3) ** 2)   # um^-2, random close packing of disks


def draw_panel(a, dens_t, meta, k, vmax, cmap, jam=True, title=None,
               xticklabels=False, yticklabels=False):
    """One map panel. Shared by the standalone figure and the application figure so the two
    cannot drift apart."""
    import matplotlib.pyplot as plt
    m, _r = masked(dens_t, meta, k)
    half = 0.5 * m.shape[0] * meta['h_nm'] * k * 1e-3
    ext = [-half, half, -half, half]
    im = a.imshow(m.T, origin='lower', extent=ext, cmap=cmap, vmin=0.0, vmax=vmax,
                  interpolation='nearest')
    for rad, ls in ((meta['r_core'], ':'), (meta['r_psmac'], '--')):
        a.add_patch(plt.Circle((0, 0), rad, fill=False, color='w', lw=0.7, ls=ls, alpha=0.85))
    # Contour at random close packing of disks. Where it appears, the description has left the
    # domain in which it means anything: no monolayer of these molecules can be that dense.
    if jam and np.nanmax(m) > RHO_JAM:
        mm = np.where(np.isfinite(m), m, 0.0)
        a.contour(np.linspace(ext[0], ext[1], mm.shape[0]),
                  np.linspace(ext[2], ext[3], mm.shape[1]),
                  mm.T, levels=[RHO_JAM], colors='w', linewidths=1.1)
        a.text(0.03, 0.03, r'$>\phi_{\rm rcp}$', transform=a.transAxes,
               fontsize=7, color='w', va='bottom')
    a.set_xticks([-5, 0, 5]); a.set_yticks([-5, 0, 5])
    lim = meta['r_disk'] * 1.13
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_aspect('equal')
    if title:
        a.set_title(title, pad=3)
    if not xticklabels:
        a.set_xticklabels([])
    if not yticklabels:
        a.set_yticklabels([])
    return im


def house_cmap():
    """The shared ramp, defined once in the Kip3 microtubule study."""
    import matplotlib.colors as mcolors
    kip3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common')   # vex-rddme/common
    sys.path.insert(0, os.path.normpath(kip3))
    import kip3_palette
    return mcolors.LinearSegmentedColormap.from_list('kip3_abyss', kip3_palette.STOPS)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--point', required=True)
    p.add_argument('--excluded', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--title', default='')
    p.add_argument('--times', default='',
                   help='comma-separated subset of the stored map times to show')
    p.add_argument('--width', type=float, default=6.69,
                   help='figure width in inches; 6.69 is the AIP double column')
    p.add_argument('--panel-letters', type=int, default=0)
    p.add_argument('--vmax', type=float, default=0.0,
                   help='colour-axis maximum in um^-2; 0 picks the 99.5th percentile of the '
                        'point-particle arm, so the arm that saturates is the one that shows it')
    p.add_argument('--cmap', default='tcr',
                   help="'tcr' uses the shared ramp from the Kip3 study's kip3_palette; "
                        'any matplotlib name also works')
    p.add_argument('--tick-size', type=float, default=2.0,
                   help='major tick length in points; ticks point inward over the '
                        'image, so the house default crowds the panel letters')
    p.add_argument('--jam', type=int, default=1,
                   help='draw a contour at the random-close-packing density of disks')
    p.add_argument('--coarsen', type=int, default=2,
                   help='block-sum factor for display; 2 gives 80 nm cells, matching '
                        'the 7093 nm^2 mesh polygon of the reference figure')
    A = p.parse_args()

    dp, mp = load(A.point)
    de, me = load(A.excluded)
    if not np.allclose(dp['map_times'], de['map_times']):
        raise SystemExit('the two runs stored maps at different times: %s vs %s'
                         % (dp['map_times'], de['map_times']))
    times = dp['map_times']
    keep = np.arange(len(times))
    if A.times:
        want = [float(x) for x in A.times.split(',')]
        keep = np.array([int(np.argmin(np.abs(times - w))) for w in want])
        if len(set(keep.tolist())) != len(keep):
            raise SystemExit('--times selected the same stored frame twice: %s' % times[keep])
    times = times[keep]
    dp_maps, de_maps = dp['maps'][keep], de['maps'][keep]
    k = A.coarsen
    dens_p = to_density(coarsen(dp_maps, k), mp, k)
    dens_e = to_density(coarsen(de_maps, k), me, k)
    vmax = A.vmax or float(np.nanpercentile(dens_p[-1], 99.5))

    import subprocess
    import matplotlib
    matplotlib.use('Agg')
    import jcp_style                                        # AIP sizes, >= 8 pt at final size
    # The manuscript is REVTeX 4.2 aip/jcp and its body is Computer Modern, so the figures are
    # set in Computer Modern through LaTeX itself. Without a usable latex on PATH matplotlib
    # falls back to mathtext SILENTLY and every glyph in the figure changes, which is why this
    # reports rather than guesses. TeX Live is mounted into the container by lmrun.sh.
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=10)
        TEX = True
    except Exception:
        TEX = False
    rc = dict(jcp_style.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in',
               'xtick.top': True, 'ytick.right': True})
    # Shorter ticks than the house default. These panels are images, not plots: the ticks point
    # inward over the data on all four sides, and at the default length the top-left pair reaches
    # far enough into the corner to touch the panel letter. 2 pt still reads at final size.
    rc.update({'xtick.major.size': A.tick_size, 'ytick.major.size': A.tick_size})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if TEX
              else {'mathtext.fontset': 'cm'})
    matplotlib.rcParams.update(rc)
    print('typeset with %s' % ('LaTeX Computer Modern' if TEX
                               else 'MATHTEXT FALLBACK - fonts will not match the manuscript'))
    import matplotlib.pyplot as plt
    cmap = house_cmap() if A.cmap == 'tcr' else A.cmap

    rho_jam = RHO_JAM
    nt = len(times)
    # Drawn at final printed size: a panel is (width - margins) / nt across, and two rows of
    # square panels set the depth. Scaling this in LaTeX would scale the type below the 8 pt AIP
    # floor, so \includegraphics must apply no scale factor.
    L, R = 0.150, 0.875
    BOT = 0.135
    TOP = 0.935 if not A.title else 0.90
    WS, HS = 0.06, 0.06
    # Panel-letter position in axes fraction, computed from the tick length actually in force so
    # that changing --tick-size moves the letter with it. Ticks point inward on all four sides and
    # one of them sits at -5 um along the top edge, directly above this corner.
    tick_ax = (matplotlib.rcParams['xtick.major.size'] / 72.0) / \
              ((R - L) * A.width / (nt + (nt - 1) * WS))
    LET_X = 0.030
    LET_Y = 1.0 - tick_ax - 0.012
    aw = (R - L) * A.width / (nt + (nt - 1) * WS)      # axes width, inches
    height = aw * (2.0 + HS) / (TOP - BOT)             # makes axes height == axes width
    fig = plt.figure(figsize=(A.width, height))
    ax = np.empty((2, nt), object)
    for ri in range(2):
        for ti in range(nt):
            ax[ri, ti] = fig.add_subplot(2, nt, ri * nt + ti + 1)
    rows = ((dens_p, mp, 'point particles'), (dens_e, me, 'volume excluded'))
    for ri, (dens, meta, label) in enumerate(rows):
        for ti in range(nt):
            im = draw_panel(ax[ri, ti], dens[ti], meta, k, vmax, cmap, jam=bool(A.jam),
                            title=('$t = %g$ s' % round(float(times[ti]), 2)) if ri == 0 else None,
                            xticklabels=(ri == 1), yticklabels=(ti == 0))
            if A.panel_letters:
                ax[ri, ti].text(LET_X, LET_Y, '(%s)' % chr(ord('a') + ri * nt + ti),
                                transform=ax[ri, ti].transAxes, va='top', ha='left', fontsize=8)
    top = TOP
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT, wspace=WS, hspace=HS)

    # The two shared axis labels and the row captions are placed from MEASURED text extents, not
    # from guessed offsets: how far the tick labels reach below the bottom row depends on the
    # font, the type size and whether LaTeX is doing the typesetting, so a fixed offset either
    # collides with them or leaves a gap. Draw once, ask each axes for the box that includes its
    # decorations, and hang the labels off that.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def tight(a):
        return a.get_tightbbox(rend).transformed(inv)

    p00 = ax[0, 0].get_position()
    p1n = ax[1, nt - 1].get_position()
    bot_edge = min(tight(ax[1, ti]).y0 for ti in range(nt))
    left_edge = min(tight(ax[ri, 0]).x0 for ri in range(2))
    PAD = 0.014
    fig.text(0.5 * (p00.x0 + p1n.x1), bot_edge - PAD, r'$x$ ($\mu$m)',
             ha='center', va='top')
    ylab_x = left_edge - PAD
    fig.text(ylab_x, 0.5 * (p1n.y0 + p00.y1), r'$y$ ($\mu$m)', rotation=90,
             ha='right', va='center')
    # Row captions sit outside the shared y label, again measured rather than assumed.
    fig.canvas.draw()
    cap_x = ylab_x - 0.030
    for ri, (_d, _m, label) in enumerate(rows):
        bb = ax[ri, 0].get_position()
        fig.text(cap_x, 0.5 * (bb.y0 + bb.y1), label, rotation=90,
                 ha='right', va='center')
    cax = fig.add_axes([0.888, 0.115, 0.016, top - 0.115])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r'TCR-pMHC surface density ($\mu$m$^{-2}$)')
    # Mark the close-packing ceiling on the colour axis when it falls inside the plotted range,
    # so the reader can see how far the densities sit from what a monolayer can hold.
    if A.jam and rho_jam <= vmax:
        cax.axhline(rho_jam, color='w', lw=0.9)
    if A.title:
        fig.suptitle(A.title, y=0.985)
    for ext_ in ('pdf', 'png'):
        # bbox_inches=None on purpose: the house RC sets savefig.bbox='tight', which would
        # re-crop the canvas and move the hand-placed labels off the positions computed above.
        fig.savefig('%s.%s' % (A.out, ext_), bbox_inches=None)
    print('wrote %s.pdf and %s.png  (colour axis 0 to %.0f um^-2; jamming at %.0f)'
          % (A.out, A.out, vmax, rho_jam))

    # The numbers a reader would want next to the panel.
    sig_um = 8.828e-3
    area = math.pi * (0.5 * sig_um) ** 2
    print('%-18s %10s %10s %12s' % ('t (s)', 'peak point', 'peak excl', 'point/excl'))
    for ti in range(nt):
        pp = float(np.nanmax(masked(dens_p[ti], mp, k)[0]))
        pe = float(np.nanmax(masked(dens_e[ti], me, k)[0]))
        print('%-18.4g %10.0f %10.0f %12s' % (times[ti], pp, pe,
              ('%.2f' % (pp / pe)) if pe > 0 else '-'))
    print('peak areal packing fraction at the last time: point %.2f, excluded %.2f'
          % (float(np.nanmax(masked(dens_p[-1], mp, k)[0])) * area,
             float(np.nanmax(masked(dens_e[-1], me, k)[0])) * area))


if __name__ == '__main__':
    main()
