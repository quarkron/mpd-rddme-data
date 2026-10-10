"""Proposed Figure 1 (fig:schematic) as TikZ, in the style of the published figure1.pdf.

    python make_figure1.py        (host venv: needs PyMuPDF for the PNGs; pdflatex from TeX Live 2026)

writes ../plots/pdf/figure1.pdf (the full figure, inside the 6.5 x 9 in text block) and figure1_{a,b,c,d}.pdf (each
panel alone, without its letter and title), the same set as PNG at 300 dpi in ../plots/png. The generated .tex sources
and LaTeX logs are kept in ../build. Edit this generator, never the .tex.

Every lattice state is described once (particles, sphere, chain) and its free-energy grid U is COMPUTED from that
content, so the U shading under a state always matches the state drawn above it:
    U(cell) = one-body potential (falls left to right) + sphere overlap + chain subunit + particle crowding.
Panels: a one timestep (diffuse -> structure -> react), b small particles, c hard-sphere crowder's fractional hop
(dilute vs crowded), d FCC chain move and chemistry among crowders. Chains sit on one checkerboard sublattice
with diagonal bonds: the 2D picture of the FCC chain lattice.
"""
import math, os, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
OUT = []
HEADINGS = True          # False: panels alone, without letter and title


def emit(s):
    OUT.append(s)


# ---------------------------------------------------------------------------------------------------- primitives
def grid(ox, oy, nx, ny, s, shell=None):
    if shell:
        for (i, j) in shell:
            emit(r'\fill[shell] (%.3f,%.3f) rectangle ++(%.3f,%.3f);' % (ox + i * s, oy + j * s, s, s))
    # TikZ snaps grid lines to multiples of the step in the CURRENT frame: draw in a frame whose origin is the corner
    emit(r'\draw[gridl, shift={(%.3f,%.3f)}] (0,0) grid[step=%.3f] (%.3f,%.3f);' % (ox, oy, s, nx * s, ny * s))
    emit(r'\draw[gridb] (%.3f,%.3f) rectangle (%.3f,%.3f);' % (ox, oy, ox + nx * s, oy + ny * s))


PENDING = []


def ugrid(ox, oy, U, s, group=0):
    PENDING.append((ox, oy, U, s, group))


def flush_u():
    """Draw the panel's U grids on ONE two-tone scale stretched over the panel's own range: states compare within a
    panel, and the full ramp is used (contrast). ulo = low U, uhi = high U."""
    for grp in sorted(set(g[4] for g in PENDING)):            # one scale per group (panel c: sphere, polymer)
        G = [g for g in PENDING if g[4] == grp]
        lo = min(min(r) for g in G for r in g[2]); hi = max(max(r) for g in G for r in g[2])
        for ox, oy, U, s, _ in G:
            _ugrid(ox, oy, [[int(round(100 * (u - lo) / (hi - lo))) for u in row] for row in U], s)
    PENDING.clear()


def _ugrid(ox, oy, U, s):
    ny, nx = len(U), len(U[0])
    for j in range(ny):
        for i in range(nx):
            v = U[j][i]   # two tones through white: sand (low) -> white -> navy (high), no muddy mixture in between
            col = 'ulo!%d!white' % (100 - 2 * v) if v < 50 else 'uhi!%d!white' % (2 * v - 100)
            emit(r'\fill[%s] (%.3f,%.3f) rectangle ++(%.3f,%.3f);' % (col, ox + i * s, oy + j * s, s, s))
    emit(r'\draw[white, line width=0.7pt, shift={(%.3f,%.3f)}] (0,0) grid[step=%.3f] (%.3f,%.3f);' % (ox, oy, s, nx * s, ny * s))
    emit(r'\draw[gridb] (%.3f,%.3f) rectangle (%.3f,%.3f);' % (ox, oy, ox + nx * s, oy + ny * s))


def sphere(ox, oy, s, c, ghost=None):
    """Sphere of diameter 2s centred at c (cell units). ghost = previous centre, drawn dashed, with the displacement."""
    if ghost:
        gx, gy = ox + ghost[0] * s, oy + ghost[1] * s
        emit(r'\draw[ink, dashed, line width=0.6pt] (%.3f,%.3f) circle (%.3f);' % (gx, gy, s))
    x, y = ox + c[0] * s, oy + c[1] * s
    emit(r'\fill[sph, draw=ink, line width=0.7pt] (%.3f,%.3f) circle (%.3f);' % (x, y, s))
    emit(r'\fill[ink] (%.3f,%.3f) circle (0.9pt);' % (x, y))
    if ghost:
        emit(r'\draw[mv] (%.3f,%.3f) -- (%.3f,%.3f);' % (gx, gy, x + 0.06 * s, y))


def chain(ox, oy, s, cells, mono=None):
    pts = [(ox + (i + .5) * s, oy + (j + .5) * s) for i, j in cells]
    for x, y in pts:
        emit(r'\fill[sub] (%.3f,%.3f) circle (%.3f);' % (x, y, 0.40 * s))
    if mono:
        x, y = ox + (mono[0] + .5) * s, oy + (mono[1] + .5) * s
        emit(r'\fill[sub!55!white, draw=sub!80!black, line width=0.4pt] (%.3f,%.3f) circle (%.3f);' % (x, y, 0.40 * s))
        emit(r'\fill[inkdim] (%.3f,%.3f) circle (1.1pt);' % (x, y))
    emit(r'\draw[ink, line width=1.3pt] ' + ' -- '.join('(%.3f,%.3f)' % p for p in pts) + ';')
    for x, y in pts:
        emit(r'\fill[ink] (%.3f,%.3f) circle (1.3pt);' % (x, y))


def particle(ox, oy, s, p):
    """p = (kind, x, y) in cell units (x, y = position inside the lattice)."""
    k, x, y = p
    r = {'A': 2.3, 'B': 2.3, 'AB': 3.4}[k]
    emit(r'\fill[sp%s] (%.3f,%.3f) circle (%.1fpt);' % (k, ox + x * s, oy + y * s, r))


def hop(ox, oy, s, x, y, dx, dy):
    emit(r'\draw[mv] (%.3f,%.3f) -- ++(%.3f,%.3f);' % (ox + x * s + 0.16 * s * dx, oy + y * s + 0.16 * s * dy,
                                                      0.62 * s * dx, 0.62 * s * dy))


def pair(ox, oy, s, p, q, pad=0.17):
    """Ellipse around exactly two reactants p, q (cell units): long axis along p-q."""
    (x1, y1), (x2, y2) = p, q
    cx, cy = ox + 0.5 * (x1 + x2) * s, oy + 0.5 * (y1 + y2) * s
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)); half = 0.5 * math.hypot(x2 - x1, y2 - y1) * s
    emit(r'\draw[ink, line width=0.8pt, rotate around={%.1f:(%.3f,%.3f)}] (%.3f,%.3f) ellipse (%.3f and %.3f);'
         % (ang, cx, cy, cx, cy, half + pad * s, 0.75 * pad * s + 0.06))


def ring(ox, oy, s, x, y, r=0.36):
    emit(r'\draw[ink, line width=0.8pt] (%.3f,%.3f) circle (%.3f);' % (ox + x * s, oy + y * s, r * s))


def harpoons(x0, x1, y, short=False, label=None, num=None):
    """Forward/backward half-arrows; short = a thinned rate (smaller, thinner pair centred in the gap)."""
    if short:
        m, h = 0.5 * (x0 + x1), 0.22 * (x1 - x0)
        x0, x1, lw, d = m - h, m + h, 0.8, 0.06
    else:
        lw, d = 1.2, 0.08
    emit(r'\draw[-{Straight Barb[left,length=5pt]}, line width=%.1fpt, ink] (%.3f,%.3f) -- (%.3f,%.3f);' % (lw, x0, y + d, x1, y + d))
    emit(r'\draw[-{Straight Barb[left,length=5pt]}, line width=%.1fpt, ink] (%.3f,%.3f) -- (%.3f,%.3f);' % (lw, x1, y - d, x0, y - d))
    if num:
        emit(r'\node[lbl, above=4pt] at (%.3f,%.3f) {\num{%s}};' % (0.5 * (x0 + x1), y, num))
    if label:
        emit(r'\node[lbl, %s] at (%.3f,%.3f) {%s};' % ('below=5pt' if num else 'above=7pt', 0.5 * (x0 + x1), y, label))


def reads(xr, yu, h, mx, ytop, label=None, band=0.45):
    """Dashed arrow from a U grid (right edge xr, bottom yu, height h) up to the step that reads it; label = what the
    step reads, written along the arrow."""
    emit(r'\draw[reads] (%.3f,%.3f) .. controls (%.3f,%.3f) .. (%.3f,%.3f);'
         % (xr + 0.08, yu + 0.6 * h, mx - 0.1, yu + h, mx, ytop))
    if label:   # beside the arrow, in the band between the lattice row and the U row
        emit(r'\node[anchor=west, align=left, font=\footnotesize, inkdim, inner sep=0pt] at (%.3f,%.3f) {%s};'
             % (mx + 0.1, yu + h + 0.5 * band, label))


def op_arrow(x0, x1, y, label, num):
    emit(r'\draw[op] (%.3f,%.3f) -- (%.3f,%.3f);' % (x0, y, x1, y))
    emit(r'\node[lbl, above=3pt] at (%.3f,%.3f) {\num{%s}};' % (0.5 * (x0 + x1), y, num))
    emit(r'\node[lbl, below=3pt] at (%.3f,%.3f) {%s};' % (0.5 * (x0 + x1), y, label))


def down(x, y0, y1):
    emit(r'\draw[gen] (%.3f,%.3f) -- (%.3f,%.3f);' % (x, y0, x, y1))


def heading(x, y, letter, text):
    if not HEADINGS:
        return
    # the letter on its own line above the title, as in the published figure
    emit(r'\node[anchor=base west, font=\fontsize{14}{16}\selectfont\bfseries, ink, inner sep=0pt] at (%.3f,%.3f) {%s};' % (x, y + 0.6, letter))
    # 14 pt here prints at 12.5 pt: the figure is placed at 0.889 scale; Figs 2 and 3 print their letters at 12-13 pt
    emit(r'\node[anchor=base west, font=\normalsize\bfseries, ink, inner sep=0pt] at (%.3f,%.3f) {%s};' % (x, y, text))


def axes(x, y, L=0.5):
    """x right, y up, z out of the page (circled dot), origin at (x, y)."""
    emit(r'\draw[-{Stealth[length=4pt]}, line width=0.7pt, ink] (%.3f,%.3f) -- ++(%.3f,0) node[right=0pt, font=\small] {$x$};' % (x, y, L))
    emit(r'\draw[-{Stealth[length=4pt]}, line width=0.7pt, ink] (%.3f,%.3f) -- ++(0,%.3f) node[above=0pt, font=\small] {$y$};' % (x, y, L))
    emit(r'\draw[ink, line width=0.6pt, fill=white] (%.3f,%.3f) circle (0.08);' % (x, y))
    emit(r'\fill[ink] (%.3f,%.3f) circle (0.025);' % (x, y))
    emit(r'\node[font=\small, ink, anchor=north east, inner sep=1pt] at (%.3f,%.3f) {$z$};' % (x - 0.06, y - 0.06))


def tag(x, y, text, anchor='east'):
    emit(r'\node[lbl, anchor=%s] at (%.3f,%.3f) {%s};' % (anchor, x, y, text))


# ---------------------------------------------------------------------------------------------------- the landscape
def overlap(c, i, j, n=12):
    """Fraction of cell (i, j) inside the sphere of radius 1 (cell units) centred at c."""
    k = 0
    for a in range(n):
        for b in range(n):
            x, y = i + (a + .5) / n, j + (b + .5) / n
            k += (x - c[0]) ** 2 + (y - c[1]) ** 2 <= 1.0
    return k / n / n


def landscape(nx, ny, parts=(), spheres=(), chain_cells=(), mono=None, slope=6.0, wsph=60.0):
    U = []
    for j in range(ny):
        row = []
        for i in range(nx):
            u = slope * (nx - 1 - i)                                        # one-body potential, drift to the right
            u += sum(wsph * overlap(c, i, j) for c in spheres)                # the sphere's excluded volume
            # a subunit fills its voxel; a free monomer of the same species and size excludes the same volume, so
            # polymerising leaves this site's U unchanged (the swap): the bond energy is the chain's own, not a site field
            u += 45 if ((i, j) in chain_cells or mono == (i, j)) else 0
            u += 14 * sum(1 for p in parts if int(p[1]) == i and int(p[2]) == j)  # crowding by the particles there
            row.append(u)
        U.append(row)
    return U


# ---------------------------------------------------------------------------------------------------- panel a
def panel_a(Y):
    heading(-0.1, Y + 0.55, 'a', 'Lattice state evolution in a single timestep')
    s, nx, ny = 0.43, 6, 4
    W, gap = nx * s, 1.75                                               # the gap holds the step word ('structure')
    X = [16.4 - (4 * W + 3 * gap) + k * (W + gap) for k in range(4)]   # right-aligned: the axes take the left margin
    yl, yu = Y - 1.95, Y - 1.95 - 0.45 - ny * s                         # lattice and U origins
    chain0 = [(0, 0), (1, 1), (0, 2)]
    chain1 = chain0 + [(1, 3)]
    p0 = [('A', 2.5, 3.45), ('B', 3.55, 3.6), ('A', 5.5, 0.5), ('B', 2.45, 1.5), ('A', 5.6, 3.5), ('B', 3.5, 0.45),
          ('A', 1.6, 2.4)]
    p1 = [('A', 3.35, 3.4), ('B', 3.6, 3.6), ('A', 5.55, 1.45), ('B', 2.5, 0.5), ('A', 5.6, 3.5), ('B', 3.5, 0.45),
          ('A', 1.6, 2.4)]
    p3 = [('AB', 3.5, 3.5), ('A', 5.55, 1.45), ('B', 2.5, 0.5), ('A', 5.6, 3.5), ('B', 3.5, 0.45), ('A', 1.6, 2.4)]
    c0, c1 = (4.0, 2.0), (4.3, 2.0)
    states = [  # parts, sphere centre, ghost, chain, monomer
        (p0, c0, None, chain0, (1, 3)),
        (p1, c0, None, chain0, (1, 3)),
        (p1, c1, c0, chain1, None),
        (p3, c1, None, chain1, None)]
    for k, (parts, c, ghost, ch, mono) in enumerate(states):
        ox = X[k]
        grid(ox, yl, nx, ny, s)
        sphere(ox, yl, s, c, ghost)
        chain(ox, yl, s, ch, mono)
        for p in parts:
            particle(ox, yl, s, p)
        down(ox + W / 2, yl - 0.08, yu + ny * s + 0.08)
        ugrid(ox, yu, landscape(nx, ny, parts, [c], ch, mono), s)
    # diffusion hops at t0, reacting pair after the structure step, growth site
    for x, y, dx, dy in ((2.5, 3.45, 1, 0), (5.5, 0.5, 0, 1), (2.45, 1.5, 0, -1)):
        hop(X[0], yl, s, x, y, dx, dy)
    pair(X[2], yl, s, (3.35, 3.4), (3.6, 3.6))
    pair(X[1], yl, s, (0.5, 2.5), (1.5, 3.5), pad=0.55)            # chain end + the monomer that joins in step 2
    for k, (lab, num) in enumerate((('diffuse', 1), ('structure', 2), ('react', 3))):
        op_arrow(X[k] + W + 0.15, X[k + 1] - 0.15, yl + ny * s / 2, lab, num)
    tag(X[0] + W / 2, yl + ny * s + 0.3, r'$t=t_0$', 'center')
    tag(X[3] + W / 2, yl + ny * s + 0.3, r'$t=t_0+\Delta t$', 'center')
    for k in range(3):
        mx = 0.5 * (X[k] + W + X[k + 1])
        emit(r'\draw[reads] (%.3f,%.3f) .. controls (%.3f,%.3f) .. (%.3f,%.3f);'
             % (X[k] + W + 0.08, yu + 0.6 * ny * s, mx - 0.1, yu + ny * s, mx, yl + ny * s / 2 - 0.62))
    tag(X[0] - 0.12, yu + ny * s / 2, r'$U$')
    axes(0.2, yu - 0.62, L=0.42)                                        # lower left corner, inside the panel border
    # key: the two-tone scale
    kx, ky = X[3] + W - 2.0, Y + 0.6
    emit(r'\shade[left color=ulo, right color=uhi, middle color=white, draw=gridc!70!ink, line width=0.5pt] (%.3f,%.3f) rectangle ++(1.4,0.2);' % (kx, ky))
    tag(kx - 0.08, ky + 0.1, r'low $U$')
    tag(kx + 1.48, ky + 0.1, 'high', 'west')
    flush_u()
    return yu - 0.95


# ---------------------------------------------------------------------------------------------------- panel b
def panel_b(Y):
    heading(-0.1, Y + 0.55, 'b', 'Reaction-diffusion for particles smaller than a subvolume')
    s, nx, ny = 0.95, 2, 1
    W, gap = nx * s, 2.3
    X0 = 0.5 * (16.5 - (3 * W + 2 * gap))                             # centred on the page
    X = [X0 + k * (W + gap) for k in range(3)]
    yl, yu = Y - 0.2 - s, Y - 0.2 - s - 0.45 - s
    st = [[('A', 0.5, 0.5), ('A', 1.22, 0.3), ('A', 1.3, 0.7), ('B', 1.55, 0.55), ('B', 1.8, 0.28)],
          [('A', 0.3, 0.35), ('A', 0.7, 0.65), ('A', 1.3, 0.7), ('B', 1.55, 0.55), ('B', 1.8, 0.28)],
          [('A', 0.3, 0.35), ('A', 0.7, 0.65), ('AB', 1.42, 0.62), ('B', 1.8, 0.28)]]
    for k, parts in enumerate(st):
        grid(X[k], yl, nx, ny, s)
        for p in parts:
            particle(X[k], yl, s, p)
        down(X[k] + W / 2, yl - 0.08, yu + s + 0.08)
        ugrid(X[k], yu, landscape(nx, ny, parts, slope=0.0), s)
    hop(X[0], yl, s, 1.22, 0.3, -1, 0)
    pair(X[1], yl, s, (1.3, 0.7), (1.55, 0.55))
    harpoons(X[0] + W + 0.2, X[1] - 0.2, yl + s / 2, label='diffuse', num=1)
    harpoons(X[1] + W + 0.2, X[2] - 0.2, yl + s / 2, label='react', num=3)
    for k in range(2):
        reads(X[k] + W, yu, s, 0.5 * (X[k] + W + X[k + 1]), yl + s / 2 - 0.62)
    tag(X[0] - 0.12, yu + s / 2, r'$U$')
    flush_u()
    return yu


# ---------------------------------------------------------------------------------------------------- panels c, d
def sequence(X, yl, W, draw_state, arrows, group, band=0.85):
    """Panel b's format for a row of states: lattice on top, its U grid below (generated), harpoons between states,
    and a dashed arrow from each U to the step that reads it. draw_state(k, x, yl) draws state k and returns its U;
    arrows = [(short, label_above, reads_label), ...] between consecutive states."""
    yu = yl - band - W                                                 # band: room for the reads labels
    for k, x in enumerate(X):
        U, s = draw_state(k, x, yl)
        down(x + W / 2, yl - 0.08, yu + W + 0.08)
        ugrid(x, yu, U, s, group=group)
    for k, (short, label, rlab) in enumerate(arrows):
        mx = 0.5 * (X[k] + W + X[k + 1])
        harpoons(X[k] + W + 0.15, X[k + 1] - 0.15, yl + W / 2, short=short, label=label)
        reads(X[k] + W, yu, W, mx, yl + W / 2 - 0.2, rlab, band)
    return yu


def panel_c(Y):
    """Hard-sphere crowder, dilute | crowded: the same fractional hop, thinned in the crowded shell."""
    heading(-0.1, Y + 0.55, 'c', 'Mobility of hard spheres larger than a subvolume')
    n, W, gap, mid = 4, 1.6, 2.1, 1.7
    s = W / n
    shell = [(i, j) for i in range(n) for j in range(n) if i in (0, n - 1) or j in (0, n - 1)]
    bath = {'dilute': [(0.4, 3.6), (3.6, 0.4)],
            'crowded': [(0.4, 3.6), (3.6, 0.4), (0.35, 0.4), (0.4, 1.6), (1.5, 0.35), (2.6, 3.65), (3.65, 2.4),
                        (3.6, 1.4), (1.4, 3.6), (0.5, 2.5), (2.5, 0.6), (3.4, 3.5)]}
    pw = 2 * W + gap
    X0 = 0.5 * (16.5 - (2 * pw + mid))
    yl = Y - 0.85 - W
    for q, name in enumerate(('dilute', 'crowded')):
        px = X0 + q * (pw + mid)
        parts = [('A', x, y) for x, y in bath[name]]

        def state(k, x, yl, parts=parts):
            c, ghost = (((2.0, 2.0), None), ((2.3, 2.0), (2.0, 2.0)))[k]
            grid(x, yl, n, n, s, shell)
            sphere(x, yl, s, c, ghost)
            for p in parts:
                particle(x, yl, s, p)
            return landscape(n, n, parts, [c], slope=0.0, wsph=22.0), s
        yu = sequence([px, px + W + gap], yl, W, state,
                      [(name == 'crowded', r'\shortstack{fractional\\move}' if q == 0 else None, r'friction\\$+\,U$')], 1)
        tag(px + W + gap / 2, yl + W + 0.55, ('less crowding, higher mobility', 'more crowding, less mobility')[q], 'center')
        tag(px - 0.12, yu + W / 2, r'$U$')
    flush_u()
    return yu


def panel_d(Y):
    """FCC chain among crowders: a move (thinned by crowding), then chemistry (U only, not thinned)."""
    heading(-0.1, Y + 0.55, 'd', 'Polymer relaxation and chemistry')
    n, W, gap = 4, 1.6, 1.7
    s = W / n
    bath = [(0.4, 0.4), (1.6, 0.4), (0.4, 1.6), (3.6, 1.6), (3.6, 3.6), (1.5, 3.6)]
    states = (([(0, 3), (1, 2), (2, 3)], (3, 0)), ([(0, 3), (1, 2), (2, 1)], (3, 0)),
              ([(0, 3), (1, 2), (2, 1), (3, 0)], None))
    rw = 3 * W + 2 * gap
    px = 0.5 * (16.5 - rw)
    yl = Y - 0.35 - W
    parts = [('A', x, y) for x, y in bath]

    def state(k, x, yl):
        cells, mono = states[k]
        grid(x, yl, n, n, s)
        chain(x, yl, s, cells, mono)
        for p in parts:
            particle(x, yl, s, p)
        return landscape(n, n, parts, (), cells, mono, slope=0.0), s
    yu = sequence([px + k * (W + gap) for k in range(3)], yl, W, state,
                  [(False, 'relax', r'friction\\$+\,U$'), (False, 'react', r'$U$ only')], 2)
    tag(px - 0.12, yu + W / 2, r'$U$')
    flush_u()
    return yu


# ---------------------------------------------------------------------------------------------------- document
PRE = r"""%% GENERATED by scripts/make_figure1.py -- edit the generator, not this file.
\documentclass[border=6pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{sansmath}
\renewcommand{\familydefault}{\sfdefault}
\sansmath
\definecolor{ink}{HTML}{1B2430}
\definecolor{inkdim}{HTML}{5D7185}
\definecolor{gridc}{HTML}{C9D3DE}
\definecolor{sph}{HTML}{D9DEE4}
\definecolor{sub}{HTML}{C8CED6}
\definecolor{shellc}{HTML}{EEF1F5}
\definecolor{uhi}{HTML}{1D3A7A}
\definecolor{ulo}{HTML}{F5DFA0}
\definecolor{spA}{HTML}{1D3A7A}
\definecolor{spB}{HTML}{A4C639}
\definecolor{spAB}{HTML}{2BB3A3}
\tikzset{
  gridl/.style={draw=gridc, line width=0.5pt},
  gridb/.style={draw=gridc!70!ink, line width=0.6pt},
  shell/.style={fill=shellc},
  op/.style={-{Stealth[length=7pt,width=5pt]}, line width=1.4pt, ink},
  mv/.style={-{Stealth[length=4pt,width=3.5pt]}, line width=0.9pt, ink},
  gen/.style={-{Stealth[length=5pt]}, line width=0.9pt, inkdim},
  lbl/.style={font=\small, ink},
  reads/.style={-{Stealth[length=5pt]}, line width=0.9pt, inkdim, dashed},
}
\newcommand{\num}[1]{\tikz[baseline=(n.base)]\node[draw=ink, circle, line width=0.6pt, inner sep=0.8pt, font=\scriptsize](n){#1};}
\begin{document}
\begin{tikzpicture}[x=0.9cm, y=0.88cm]
"""
POST = r"""\end{tikzpicture}
\end{document}
"""


def full():
    rule = lambda y: emit(r'\draw[gridc, line width=0.6pt] (-0.1,%.3f) -- (16.6,%.3f);' % (y, y))
    y = panel_a(0.0)
    G = 2.0                                                # rule -> next panel's title baseline (letter line above it)
    rule(y - 0.1)
    y = panel_b(y - G) - 0.3
    rule(y - 0.1)
    y = panel_c(y - G) - 0.3
    rule(y - 0.1)
    panel_d(y - G)


def build(name, draw):
    """Generate build/<name>.tex, compile it, copy the PDF to plots/pdf and render plots/png (300 dpi)."""
    OUT.clear(); PENDING.clear()
    draw()
    bdir = os.path.join(ROOT, 'build'); os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, name + '.tex'), 'w') as f:
        f.write(PRE + '\n'.join(OUT) + '\n' + POST)
    env = dict(os.environ)                      # pdflatex from PATH (TeX Live 2026 used for the published figure)
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'], cwd=bdir, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('pdflatex failed on %s: see %s' % (name, os.path.join(bdir, name + '.log')))
    pdf = os.path.join(ROOT, 'plots', 'pdf', name + '.pdf')
    shutil.copyfile(os.path.join(bdir, name + '.pdf'), pdf)
    import fitz                                            # PyMuPDF
    pg = fitz.open(pdf)[0]
    pg.get_pixmap(dpi=300).save(os.path.join(ROOT, 'plots', 'png', name + '.png'))
    print('%-12s %.2f x %.2f in' % (name, pg.rect.width / 72, pg.rect.height / 72))


def main():
    global HEADINGS
    for d in ('pdf', 'png'):
        os.makedirs(os.path.join(ROOT, 'plots', d), exist_ok=True)
    HEADINGS = True
    build('figure1', full)
    HEADINGS = False
    for letter, fn in (('a', panel_a), ('b', panel_b), ('c', panel_c), ('d', panel_d)):
        build('figure1_' + letter, lambda fn=fn: fn(0.0))


if __name__ == '__main__':
    main()
