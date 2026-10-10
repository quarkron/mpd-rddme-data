"""The setup cartoon for the sinusoidal-field reaction figure: one voxel, the reaction, the field.

    draw_scheme(ax)   -- ax is given xlim 0..113, ylim 0..26 and equal aspect

Three things have to be said before any of the measurements mean anything, and each is drawn
rather than described:

    the subvolume    a 10 nm voxel and what it holds. The run carries about three of each
                     reactant per voxel, which is what sets both the exclusion the method
                     computes and the shot noise every panel of the figure carries; the cartoon
                     is a section through that voxel, so it shows the two or three of them a
                     plane actually cuts. The spheres are placed by rejection, never overlapping:
                     this is the panel that says the particles are hard.
    the reaction     A + B <=> C at fixed rate constants, with the product sized so that
                     sigma_C^3 = sigma_A^3 + sigma_B^3 -- the total occupied volume is therefore
                     conserved by the reaction, and any density change seen later is the field's
                     doing and not the chemistry's;
    the field        a static sinusoid along one axis, four periods across the 320 nm box, its
                     period eight voxels wide so the lattice resolves it.

Spheres are drawn at their true relative diameters on the voxel's own scale, so the reader can
see that a reactant is a third of a voxel across; nothing here is to a second scale except the
field strip, which is the whole box and is marked as such.
"""
import numpy as np

W, H = 113.0, 20.5            # the cartoon's own coordinate system, equal aspect
HV = 12.0                     # a voxel, in those units: 10 nm
NM = HV / 10.0                # units per nm, for the spheres
BOX_X0, BOX_X1 = 52.0, 112.0  # the 320 nm field strip
NVOX = 32                     # voxels across the box
NPER = 8                      # voxels per period

SIG = {'A': 3.0, 'B': 3.0, 'C': 3.78}
COL = {'A': '#17509b', 'B': '#41c5bd', 'C': '#a3dda0'}


def _sphere(ax, xy, sigma, name, ink, z=3):
    """A particle at its true diameter, with a highlight so it reads as a sphere not a disc."""
    from matplotlib.patches import Circle
    r = 0.5 * sigma * NM
    ax.add_patch(Circle(xy, r, facecolor=COL[name], edgecolor=ink, lw=0.4, zorder=z))
    ax.add_patch(Circle((xy[0] - 0.28 * r, xy[1] + 0.3 * r), 0.34 * r, facecolor='white',
                        edgecolor='none', alpha=0.45, zorder=z + 0.1))


def draw_scheme(ax, tex=True):
    from matplotlib.patches import Rectangle, FancyArrow
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))
    import kip3_palette as P

    ink, dim, cage = P.INK, P.INK_DIM, P.CAGE
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')
    m = (lambda s: s) if tex else (lambda s: s)

    # ---- the subvolume ---------------------------------------------------------------------
    vx, vy = 3.0, 5.2
    ax.add_patch(Rectangle((vx, vy), HV, HV, facecolor='#f2f6fa', edgecolor=cage, lw=0.7,
                           zorder=1))
    # Rejection sampling, not free placement: this is the figure that tells the reader the
    # particles are hard, and a cartoon with two of them overlapping says the opposite.
    rng = np.random.default_rng(11)
    placed = []
    for name, n in (('C', 1), ('A', 2), ('B', 2)):
        r = 0.5 * SIG[name] * NM
        for _ in range(n):
            for _try in range(4000):
                q = (vx + r + (HV - 2 * r) * rng.random(), vy + r + (HV - 2 * r) * rng.random())
                if all(np.hypot(q[0] - o[0], q[1] - o[1]) >= r + orad for o, orad in placed):
                    placed.append((q, r))
                    _sphere(ax, q, SIG[name], name, ink)
                    break
            else:
                raise RuntimeError('no room left in the voxel cartoon for a %s' % name)
    ax.annotate('', (vx, vy - 1.3), (vx + HV, vy - 1.3),
                arrowprops=dict(arrowstyle='<->', color=dim, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(vx + 0.5 * HV, vy - 2.3, r'$h = 10$ nm', ha='center', va='top', color=ink, fontsize=8)
    ax.text(vx + 0.5 * HV, vy + HV + 0.8, r'$\eta = 0.088$', ha='center', va='bottom',
            color=ink, fontsize=8)

    # ---- the reaction ----------------------------------------------------------------------
    # Species names sit ABOVE their spheres and the two rate constants below the whole line:
    # with both under the spheres the k_off string runs straight through the B and C labels.
    y0 = 11.2
    xs = [24.0, 31.5, 45.0]
    for name, x in zip(('A', 'B', 'C'), xs):
        _sphere(ax, (x, y0), SIG[name], name, ink)
        ax.text(x, y0 + 0.5 * SIG[name] * NM + 0.9, r'$\mathsf{%s}$' % name, ha='center',
                va='bottom', color=ink, fontsize=8.5)
    ax.text(0.5 * (xs[0] + xs[1]), y0, '+', ha='center', va='center', color=ink, fontsize=9)
    ax.annotate('', (xs[2] - 3.4, y0 + 1.1), (xs[1] + 2.6, y0 + 1.1),
                arrowprops=dict(arrowstyle='->', color=ink, lw=0.8, shrinkA=0, shrinkB=0))
    ax.annotate('', (xs[1] + 2.6, y0 - 1.1), (xs[2] - 3.4, y0 - 1.1),
                arrowprops=dict(arrowstyle='->', color=ink, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(34.0, y0 - 3.8, r'$K = 200$ nm$^3$', ha='center', va='top', color=ink, fontsize=7.5)
    ax.text(34.0, y0 - 6.0, r'$k_{\mathrm{off}} = 5\times10^{5}$ s$^{-1}$', ha='center',
            va='top', color=ink, fontsize=7.5)
    ax.text(34.0, y0 - 8.2, r'$D = 0.1\ \mu$m$^2$/s (each species)', ha='center', va='top', color=ink,
            fontsize=7.5)
    ax.text(34.0, vy + HV + 0.8, r'$\sigma_\mathsf{C}^{3} = \sigma_\mathsf{A}^{3}'
            r' + \sigma_\mathsf{B}^{3}$', ha='center', va='bottom', color=ink, fontsize=8)

    # ---- the field ---------------------------------------------------------------------------
    L = BOX_X1 - BOX_X0
    strip_y, strip_h = 4.0, 3.3
    ax.add_patch(Rectangle((BOX_X0, strip_y), L, strip_h, facecolor='#eef3f8', edgecolor=cage,
                           lw=0.7, zorder=1))
    for i in range(NVOX + 1):
        x = BOX_X0 + L * i / NVOX
        ax.plot([x, x], [strip_y, strip_y + strip_h], color=cage, lw=0.25, alpha=0.7, zorder=2)
    ax.text(BOX_X0, strip_y - 0.8, '0', ha='center', va='top', color=dim, fontsize=7.5)
    ax.text(BOX_X1, strip_y - 0.8, '320 nm', ha='center', va='top', color=dim, fontsize=7.5)

    amp, mid = 3.5, 14.0
    xx = np.linspace(0, 1, 400)
    ax.plot(BOX_X0 + L * xx, mid + amp * np.sin(2 * np.pi * NVOX / NPER * xx),
            color=P.ARM['exc'], lw=1.3, zorder=3)
    for i in range(NVOX + 1):
        x = BOX_X0 + L * i / NVOX
        ax.plot([x], [mid + amp * np.sin(2 * np.pi * i / NPER)], marker='o', ms=1.7,
                color=P.ARM['exc'], zorder=4)
    # gamma above the first crest, lambda below between two troughs: the two annotations never
    # share a band of the cartoon, and neither crosses the curve.
    xc = BOX_X0 + L * 2.0 / NVOX
    ax.annotate('', (xc, mid), (xc, mid + amp),
                arrowprops=dict(arrowstyle='<->', color=dim, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(xc, mid + amp + 0.7, r'$\gamma = 0.3$ kT', ha='center', va='bottom', color=ink,
            fontsize=7.5)
    yl = mid - amp - 0.9
    ax.annotate('', (BOX_X0 + L * 6.0 / NVOX, yl), (BOX_X0 + L * 14.0 / NVOX, yl),
                arrowprops=dict(arrowstyle='<->', color=dim, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(BOX_X0 + L * 10.0 / NVOX, yl - 0.6, r'$\lambda = 80$ nm $= 8h$', ha='center',
            va='top', color=ink, fontsize=7.5)
    ax.text(BOX_X1, H - 0.3, r'$\Phi_s(x) = \gamma\sin(2\pi x/\lambda)$', ha='right',
            va='top', color=ink, fontsize=8.5)
