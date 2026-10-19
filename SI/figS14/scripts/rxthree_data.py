"""Shared analysis for the ternary relaxation figures: the main-text figure (rxthree_relax_fig.py) and
the cross-species correlation figure in the supplement (rxthree_rho_fig.py).

Both figures score the same three arms over the same common tau grid, so the pooling lives here rather
than in either of them. That is not only tidiness: the correlation panel moved to the supplement while
the divergence panels stayed in the main text, and if the two scripts pooled independently a reader
comparing them would have no guarantee that "the same run at the same time" meant the same thing in
both. One `prepare()` and they cannot drift.

DATA
    readdy_ref/out_three_relax/rt3_s*.npz              Brownian reference, 40 seeds (readdy_three.py)
    mpd_crddme/relax3_out/lat3_{model,point}_s*.npz    MPD-RDDME and the point-particle limit
Both are symlinks into /raid/racda/campaigns; see studies/FIGURES.md.

CARES
    Every run is placed on the common grid by nearest frame, because adaptive tau gives each lattice
    seed its own physical grid. C's displacement is unwrapped across the periodic box. The correlation
    is taken over ALL voxels, so the sign change the slab start produces is visible.
"""
import glob, json, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'common'))   # vex-rddme/common: jcp_style, kip3_palette
sys.path.insert(0, HERE)
import jcp_style
import kip3_palette as P
import rxthree_scene3d as S3

# Relative to the study, not absolute: inside the lm2.6 container the repository is mounted at /ftsz,
# so an absolute host path does not exist there and the save fails after every panel has been drawn.
FIGD = os.environ.get('RXTHREE_FIGD', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')))   # vex-rddme: plots/{pdf,png}
NMAX_A, NMAX_B = 40, 12
H = 10.0; D_A = 1e5; D_C = 1e4     # nm, nm^2/s (readdy_three.py --d-a, ternary_relax.py D_UM2)
SIG = {'A': 2.0, 'B': 4.0, 'C': 20.0}

# ★6.5 in is the 468 pt text width of the manuscript's single-column preprint layout, NOT
# jcp_style.FULL (6.69 in, the two-column reprint width). Drawn at 6.69 a figure is scaled by 0.97 on
# the page and every 8 pt label falls under the AIP floor.
TEXTW = 6.5

# Series colours from the Kip3 palette, assigned by meaning rather than by position: the lattice arm
# that carries exclusion takes the palette's exclusion colour, the point-particle arm takes its point
# colour, and the Brownian reference takes the neutral slate the palette reserves for a reference. It
# is keyed as BD, not as ReaDDy: the comparison is against Brownian dynamics as a method, and ReaDDy
# is the code that ran it. The body text names the code.
REF = P.ARM['ref']
ARMS = (('model', 'MPD-RDDME', P.ARM['exc'], 's--'),
        ('point', 'point particles', P.ARM['pt'], '^:'))
COL = {'A': P.STOPS[3], 'B': P.STOPS[1], 'C': S3.BODY}       # cartoon species, off the same ramp


def load(paths):
    out = []
    for f in sorted(paths):
        d = np.load(f, allow_pickle=True); m = json.loads(str(d['meta']))
        n = int(round(m['L'] / m['h'])) if 'L' in m else int(m['N'])
        posC = np.asarray(d['posC'], float)
        if 'L' in m: posC = posC / m['h']                    # ReaDDy: nm box coords -> voxel coords
        out.append(dict(tau=np.asarray(d['tau'], float), A=d['occA'], B=d['occB'], posC=posC, n=n,
                        meta=m, path=f))
    return out


def nearest(run, grid):
    return [int(np.argmin(np.abs(run['tau'] - g))) for g in grid]


def pooled(runs, grid, key):
    return np.concatenate([r[key][nearest(r, grid)] for r in runs], axis=1)   # (ngrid, seeds*voxels)


def hist(c, nmax):
    P_ = np.zeros((c.shape[0], nmax + 1))
    for i in range(c.shape[0]):
        P_[i] = np.bincount(np.clip(c[i], 0, nmax), minlength=nmax + 1)[:nmax + 1]
    return P_ / P_.sum(1, keepdims=True)


def kl(p, q, qn):
    q = np.maximum(q, 1.0 / qn); q = q / q.sum(); m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def rho(a, b):
    a = a.astype(float); b = b.astype(float); sa, sb = a.std(1), b.std(1)
    cov = ((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).mean(1)
    return np.where((sa > 0) & (sb > 0), cov / (sa * sb + 1e-300), np.nan)


def msd_raw(runs, grid):
    vals = []
    for r in runs:
        n = r['n']; p = r['posC'].copy()
        dp = np.diff(p, axis=0); dp -= n * np.round(dp / n)
        u = np.concatenate([p[:1], p[:1] + np.cumsum(dp, axis=0)])
        vals.append(((u[nearest(r, grid)] - u[0]) ** 2).sum(-1) * H * H)
    v = np.concatenate(vals, axis=1)
    return v.mean(1), v.std(1, ddof=1) / math.sqrt(v.shape[1])


def initial_configuration(seed=0):
    """The reference's tau = 0 configuration for `seed`, regenerated with the driver's own initializer
    (deterministic)."""
    src = open(os.path.join(HERE, 'readdy_three.py')).read()
    fn = src[src.index('def slab_init_mix('):src.index('\n\n\nsystem = readdy.ReactionDiffusionSystem')]
    ns = {'np': np, 'L': 80.0, 'sig': SIG, 'rng': np.random.default_rng(seed)}
    exec(fn, ns)
    V = 80.0 ** 3
    N = {k: int(round(0.07 * V / (np.pi / 6 * SIG[k] ** 3))) for k in 'AB'}
    cpos = np.array([[60.0, 40.0, 40.0]])
    pos, spec = ns['slab_init_mix']([N['A'], N['B']], [SIG['A'], SIG['B']], 0.5, cpos)
    return pos, spec, cpos


def initial_lattice(meta):
    """The lattice arm's tau = 0 state, rebuilt from the run's own initializer.

    WHY THIS IS NOT JUST FRAME 0. The solver's hook fires after the first `per_frame` steps, so the
    earliest recorded frame is tau = 0.006, by which point 4191 particles have already crossed into
    the empty half. Drawing that and calling it the start would show a slab that is not quite a slab.

    The start is reproducible exactly, though, because ternary_relax.py builds it in pure numpy: one
    draw, rng.choice on the reference's own tau = 0 joint (n_A, n_B) histogram, into the filled
    stripes x % period < duty*period, with `rng` seeded by the run's seed and untouched before that
    draw. The run records its seed and the path to the histogram in its own metadata, so what is
    rebuilt here uses the run's inputs and nothing else. CHECKED: the reconstruction's particle
    totals, 548024 of A and 68408 of B, equal frame 0's exactly -- the solver conserves mass, so
    matching totals mean the same draw -- while its empty half is exactly zero and its filled half
    differs from frame 0 by 1.1 particles per voxel out of 33, which is those first steps of
    diffusion.

    Returns the eta field of the same 80 nm cell S3.cell_of() cuts, and the bodies' starting centres.
    """
    N = int(meta['N']); per = int(meta['period']); duty = float(meta['duty'])
    rng = np.random.default_rng(int(meta['seed']))
    J = np.load(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'main', 'fig2', 'data'), 'readdy_ref', 'out_three_relax', os.path.basename(meta['seed_hist']))).astype(float); J = J / J.sum()   # vex-rddme: local copy of the stored absolute path
    xs = np.arange(N); live = xs[(xs % per) < int(round(duty * per))]
    flat = rng.choice(J.size, size=len(live) * N * N, p=J.ravel())
    cA, cB = np.unravel_index(flat, J.shape)
    A0 = np.zeros((N, N, N)); B0 = np.zeros((N, N, N))
    A0[live] = cA.reshape(len(live), N, N); B0[live] = cB.reshape(len(live), N, N)
    cell = S3.CELL
    eta0 = S3.eta_of(A0[:cell, :cell, :cell], B0[:cell, :cell, :cell])
    pc = np.asarray(meta['centers'], float)
    inside = np.all((pc >= -0.5) & (pc < cell - 0.5), axis=1)
    return eta0, pc[inside]


def style(sans=False):
    """House rc, with usetex when a latex binary is on PATH; sans=True for the sans-serif variant.

    jcp_style alone leaves text.usetex off, so every label renders in DejaVu Serif mathtext while the
    rest of the manuscript renders in Computer Modern -- a silent fallback that changes every font in
    the figure and nothing else. Probe for latex and say which way it went."""
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5); tex = True
    except Exception:
        tex = False
    rc = dict(jcp_style.RC)
    rc.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
    rc.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if tex
              else {'mathtext.fontset': 'cm'})
    if sans:
        rc = jcp_style.sans(rc, tex)          # text AND math; see jcp_style.sans
    if not tex:
        print('  WARNING: no latex on PATH; type will not match the other figures')
    return rc


def prepare():
    """Load every arm, build the common tau grid, and reduce: marginals, divergences, correlation and
    the body's displacement. Returns one dict; both figures read from it and neither re-pools."""
    rd = load(glob.glob(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'main', 'fig2', 'data'), 'readdy_ref/out_three_relax/rt3_s*.npz')))
    lat = {a[0]: load(glob.glob(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'main', 'fig2', 'data'), 'mpd_crddme/relax3_out/lat3_%s_s*.npz' % a[0])))
           for a in ARMS}
    arms = [a for a in ARMS if lat[a[0]]]
    if not rd or not arms:
        sys.exit('need the Brownian reference and at least one lattice arm')
    print('BD %d seeds; lattice: %s'
          % (len(rd), ', '.join('%s %d' % (a[0], len(lat[a[0]])) for a in arms)))

    dtau_lat = max(np.diff(r['tau']).min() for a in arms for r in lat[a[0]])
    tmax = min(min(r['tau'][-1] for r in lat[a[0]]) for a in arms)
    grid = np.array([0.0] + [t for t in rd[0]['tau'] if dtau_lat <= t <= tmax])
    print('common grid: %d times, tau 0..%.3g (lattice frame spacing %.3g)'
          % (len(grid), grid[-1], dtau_lat))

    bA, bB = pooled(rd, grid, 'A'), pooled(rd, grid, 'B')
    PbA, PbB = hist(bA, NMAX_A), hist(bB, NMAX_B)
    rb = rho(bA, bB)
    rb_seeds = np.array([rho(pooled([r], grid, 'A'), pooled([r], grid, 'B')) for r in rd])
    rb_sem = rb_seeds.std(0, ddof=1) / math.sqrt(len(rd))
    mr_b, mr_b_e = msd_raw(rd, grid)
    L = {}
    for key, lbl, col, sty in arms:
        a, b = pooled(lat[key], grid, 'A'), pooled(lat[key], grid, 'B'); NQ = a.shape[1]
        PA, PB = hist(a, NMAX_A), hist(b, NMAX_B)
        L[key] = dict(PA=PA, PB=PB, rho=rho(a, b), msdraw=msd_raw(lat[key], grid)[0],
                      klA=[kl(PbA[i], PA[i], NQ) for i in range(len(grid))],
                      klB=[kl(PbB[i], PB[i], NQ) for i in range(len(grid))])
    return dict(rd=rd, lat=lat, arms=arms, grid=grid, PbA=PbA, PbB=PbB, rb=rb, rb_sem=rb_sem,
                mr_b=mr_b, mr_b_e=mr_b_e, L=L)


def report(D):
    """The table both figures print, so either one run alone still gives the numbers the text quotes."""
    grid, L, arms, rb = D['grid'], D['L'], D['arms'], D['rb']
    print('%7s | %s | %8s | %s' % ('tau', ' '.join('KL_A %-7s' % a[0][:7] for a in arms), 'rho BD',
                                   ' '.join('rho %-7s' % a[0][:7] for a in arms)))
    for i in range(0, len(grid), max(1, len(grid) // 8)):
        print('%7.3g | %s | %+8.3f | %s'
              % (grid[i], ' '.join('%12.4f' % L[a[0]]['klA'][i] for a in arms), rb[i],
                 ' '.join('%+11.3f' % L[a[0]]['rho'][i] for a in arms)))
    print('C MSD final (nm^2): free %.0f | BD %.0f | %s'
          % (6 * D_C * grid[-1] * H * H / D_A, D['mr_b'][-1],
             ' '.join('%s %.0f' % (a[0], L[a[0]]['msdraw'][-1]) for a in arms)))
