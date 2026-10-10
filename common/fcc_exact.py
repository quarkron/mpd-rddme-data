#!/usr/bin/env python3
"""Exact statistics and exact sampling of the FCC lattice chain.

The chain is a sequence of bonds drawn from the twelve edge diagonals of the cubic grid, all of
length sqrt(2) voxels, with a bend energy kappa*(1-cos theta) at each interior joint and no bond
term (monodisperse, so the bond energy is a constant that cancels). That bond set is the
nearest-neighbour set of an FCC sublattice: every edge-diagonal step preserves the parity of
x+y+z.

WHY THIS MODULE EXISTS. Two things the Monte Carlo sampler cannot do.

(1) EXACT REFERENCE VALUES. The octahedral group is transitive on the twelve directions, so every
    row sum of the transfer matrix is the same number z, the uniform vector is its leading
    eigenvector, and the turn angles come out independent and identically distributed. The whole
    chain is then closed form: <cos theta> = z1/z is a ratio of five exponentials, the tangent
    correlation is a single exponential 2 r^n with no subleading modes, and <R^2> at any FINITE
    chain length follows from a geometric sum. No fitted parameter at any M.

(2) EXACT CONFIGURATIONS. Equal row sums also make the backward messages constant, so the finite
    chain with free ends is exactly a first-order Markov chain in bond direction: draw the first
    bond uniformly from the twelve, each later one from B[a]/z. That is an independent draw from
    the phantom ensemble in O(M) work, with no burn-in. And because the self-avoiding weight is the
    phantom weight times an indicator on the sites being distinct, a draw that happens not to
    intersect itself is an EXACT sample of the self-avoiding chain. Rejection is cheap over the
    range that matters: acceptance is 0.98 at kappa 6 and 64 subunits, 0.32 at kappa 2, 0.024 at
    kappa 1.

    That is what `seed()` is for. The local move set equilibrates a straight or uniform-walk seed
    only below a stiffness ceiling that falls with chain length, so seeding from the exact
    distribution replaces the equilibration problem with a stationarity check.

Everything is in VOXEL units; multiply by the voxel edge for physical length. Direction indices are
the engine's own L26 numbering (chain_geometry.VECS), so `seed()` output drops straight into
fixed_chain_harness.seed_chain.

    python3 fcc_exact.py --check          # closed form against the eigensolver, and the sampler
"""
import argparse

import numpy as np

import chain_geometry as G

# ---------------------------------------------------------------------------------------------
# The bond set: the twelve edge diagonals, in the engine's L26 direction numbering.
# ---------------------------------------------------------------------------------------------
EDGE = np.flatnonzero(np.isclose(np.linalg.norm(G.VECS, axis=1), np.sqrt(2.0)))
VEC = G.VECS[EDGE].astype(np.float64)                 # (12, 3) bond vectors
COS = np.clip(VEC @ VEC.T / 2.0, -1.0, 1.0)           # all lengths sqrt(2), so dot/2 is the cosine
BOND = float(np.sqrt(2.0))                            # bond length in voxels
Z_COORD = len(EDGE)

if Z_COORD != 12:
    raise RuntimeError("expected 12 edge diagonals, found %d" % Z_COORD)


def transfer(kappa):
    """The 12x12 joint weight B[a,b] = exp(-kappa (1 - u_a . u_b))."""
    return np.exp(-float(kappa) * (1.0 - COS))


def _exp_terms(kappa):
    k = float(kappa)
    return np.exp(-k / 2.0), np.exp(-k), np.exp(-1.5 * k), np.exp(-2.0 * k)


def z_partition(kappa):
    """One-joint partition function z = sum_theta f_theta exp(-kappa(1-cos theta)).

    The angle multiset seen from any edge diagonal is 1 collinear, 4 at 60, 2 at 90, 4 at 120 and
    1 antiparallel, so z is also every row sum of B.
    """
    e1, e2, e3, e4 = _exp_terms(kappa)
    return 1.0 + 4.0 * e1 + 2.0 * e2 + 4.0 * e3 + e4


def z_cos(kappa):
    """sum_theta f_theta cos(theta) exp(-kappa(1-cos theta)), the numerator of <cos theta>."""
    e1, e2, e3, e4 = _exp_terms(kappa)
    return 1.0 + 2.0 * e1 - 2.0 * e3 - e4          # the 90 degree term carries cos = 0


def ratio(kappa):
    """r = <cos theta>, which on this bond set is also the tangent correlation decay ratio.

    The two coincide because every bond has the same length; on a mixed bond set they do not, and
    comparing a measured one-step cosine against a fitted decay ratio is then the wrong comparison.
    """
    return z_cos(kappa) / z_partition(kappa)


def lp_over_b(kappa):
    """Persistence length in bonds. Infinite at kappa where r rounds to 1 in double precision."""
    r = ratio(kappa)
    if r <= 0.0:
        return 0.0                                  # r = 0 exactly at kappa = 0: an ideal walk
    if r >= 1.0:
        return np.inf
    return -1.0 / np.log(r)


def persistence_length(kappa):
    """Persistence length in voxels."""
    return BOND * lp_over_b(kappa)


def mean_square_r(kappa, m_bonds):
    """<R^2> of the PHANTOM chain of m bonds, in voxels squared, exact at every m.

    From R^2 = sum_i b_i^2 + 2 sum_{i<j} b_i.b_j with <b_0.b_n> = 2 r^n and M-n ordered pairs at
    separation n, summed in closed form.
    """
    m = int(m_bonds)
    if m <= 0:
        return 0.0
    r = ratio(kappa)
    if r == 0.0:
        return 2.0 * m                              # ideal random walk
    tail = r * (m * (1.0 - r) - (1.0 - r ** m)) / (1.0 - r) ** 2
    return 2.0 * (m + 2.0 * tail)


# ---------------------------------------------------------------------------------------------
# Exact sampling.
# ---------------------------------------------------------------------------------------------
def markov_words(kappa, m_bonds, n_draws, rng):
    """n_draws independent exact phantom chains, as (n_draws, m_bonds) indices into EDGE.

    First bond uniform over the twelve, each later one from B[a]/z. Exact at every finite length
    because the backward messages are constant, so there are no end corrections to apply.
    """
    m, n = int(m_bonds), int(n_draws)
    cdf = np.cumsum(transfer(kappa) / z_partition(kappa), axis=1)
    w = np.empty((n, m), np.int64)
    w[:, 0] = rng.integers(0, Z_COORD, n)
    for j in range(1, m):
        u = rng.random(n)
        w[:, j] = (cdf[w[:, j - 1]] < u[:, None]).sum(axis=1)
    return np.minimum(w, Z_COORD - 1)               # guard the open right edge of the last bin


def words_to_sites(words):
    """(n, m) bond words -> (n, m+1, 3) integer site coordinates starting at the origin."""
    steps = VEC[words]
    return np.concatenate([np.zeros((len(words), 1, 3)), np.cumsum(steps, axis=1)], axis=1)


def self_avoiding_mask(sites):
    """Which of the sampled chains occupy m+1 distinct sites."""
    n, npts, _ = sites.shape
    span = 4 * npts + 4
    key = ((sites[:, :, 0] + npts) * span + (sites[:, :, 1] + npts)) * span + (sites[:, :, 2] + npts)
    key = np.sort(key, axis=1)
    return (np.diff(key, axis=1) != 0).all(axis=1)


def sample(kappa, m_bonds, n_draws, rng, self_avoiding=True):
    """Exact configurations and their squared end-to-end distances.

    Returns (r2, sites, keep). With self_avoiding the keep mask selects exact samples of the
    self-avoiding chain; the phantom weight times the no-overlap indicator IS the self-avoiding
    weight, so a surviving draw needs no reweighting.
    """
    sites = words_to_sites(markov_words(kappa, m_bonds, n_draws, rng))
    r2 = (sites[:, -1] ** 2).sum(axis=1)
    keep = self_avoiding_mask(sites) if self_avoiding else np.ones(len(sites), bool)
    return r2, sites, keep


def required_extent(kappa, n_subunits, rng, batch=4096, quantile=0.999):
    """Per-axis span a lattice must hold so that essentially every exact configuration fits.

    Sizing the box against the chain rather than the chain against the box is not fastidiousness.
    Keeping only the configurations that fit a box selects the compact tail, and at a stiffness
    where the typical chain is longer than the box that tail is nothing like the ensemble: a 64
    bond chain at kappa 6 has <R^2> = 2325 voxels squared, and the ones that fit a 32 cube have
    R^2 near 80. A box that truncates the seed truncates the run in it too, so this is reported
    rather than worked around.
    """
    m = int(n_subunits) - 1
    sites = words_to_sites(markov_words(kappa, m, int(batch), rng))
    ok = self_avoiding_mask(sites)
    if not ok.any():
        raise RuntimeError("no self-avoiding draw in %d at kappa %g, n %d" % (batch, kappa, n_subunits))
    span = (sites[ok].max(axis=1) - sites[ok].min(axis=1)).max(axis=1)
    return float(np.quantile(span, quantile))


def seed(n_subunits, kappa, lx, ly, lz, rng, margin=3, batch=4096, tries=64):
    """One exact self-avoiding configuration, centred in the lattice.

    Returns (pos, dirs) in the format fixed_chain_harness.seed_chain expects: integer sites and
    engine L26 direction indices.

    Raises rather than falling back to a uniform walk if no configuration is found, and raises
    rather than selecting from the compact tail if the lattice is too small to hold the ensemble.
    A silent fallback in either case would put a non-equilibrium seed into a run whose whole point
    is that the seed IS the equilibrium distribution.
    """
    m = int(n_subunits) - 1
    if m < 1:
        raise ValueError("need at least two subunits")
    room = min(lx, ly, lz) - 2 * margin - 1
    need = required_extent(kappa, n_subunits, rng, batch=batch)
    if room < need:
        side = int(np.ceil(need)) + 2 * margin + 1
        raise RuntimeError(
            "lattice %dx%dx%d with margin %d leaves %d voxels, but %d subunits at kappa %g span "
            "%.1f at the 99.9th percentile; keeping only what fits would seed from the compact "
            "tail. Use at least %d voxels per side."
            % (lx, ly, lz, margin, room, n_subunits, kappa, need, side))
    for _ in range(int(tries)):
        words = markov_words(kappa, m, int(batch), rng)
        sites = words_to_sites(words)
        ok = self_avoiding_mask(sites)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        spans = (sites[idx].max(axis=1) - sites[idx].min(axis=1)).max(axis=1)
        fits = idx[spans <= room]
        if not len(fits):
            continue
        i = int(rng.choice(fits))               # uniform among fitters, not the first one found
        chosen = sites[i]
        lo = np.floor(chosen.min(axis=0)).astype(np.int64)
        extent = np.ceil(chosen.max(axis=0)).astype(np.int64) - lo
        origin = (np.array([lx, ly, lz], np.int64) - extent) // 2
        pos = (chosen - lo).astype(np.int64) + origin
        dirs = EDGE[words[i]].astype(np.int64)
        return pos, dirs
    raise RuntimeError(
        "no exact self-avoiding FCC configuration of %d subunits in %d batches of %d at kappa %g; "
        "acceptance falls with length and rises with stiffness, so a long flexible chain needs a "
        "Rosenbluth-weighted grower rather than rejection"
        % (n_subunits, tries, batch, kappa))


def seed_many(k_chains, n_subunits, kappa, lx, ly, lz, rng, margin=3, gap=2):
    """K exact configurations placed one per cell of a regular grid, so none overlap at t=0.

    K bodies rather than K runs, because the structure kernels never see the simulation seed
    (struct_step_seed is keyed on the timestep counter) while they DO mix the complex id into the
    move hash. One body therefore walks the same path in every run and gives no ensemble; K bodies
    in one box give K streams.

    The cell side is the 99.9th percentile chain span plus `gap`, so the seed is collision-free with
    room to spare. The bodies then diffuse and may meet later; that is a dilute-solution correction
    of order the chain volume fraction, which the caller should report rather than assume away.
    """
    k = int(k_chains)
    span = required_extent(kappa, n_subunits, rng)
    cell = int(np.ceil(span)) + int(gap)
    grid = np.array([(lx - 2 * margin) // cell, (ly - 2 * margin) // cell, (lz - 2 * margin) // cell])
    if grid.min() < 1 or int(np.prod(grid)) < k:
        raise RuntimeError(
            "%d chains of %d subunits at kappa %g need %d^3 cells of %d voxels inside a "
            "%dx%dx%d lattice with margin %d, which holds %d; enlarge the lattice or lower K"
            % (k, n_subunits, kappa, cell, cell, lx, ly, lz, margin, max(0, int(np.prod(grid)))))
    cells = [(i, j, m) for i in range(grid[0]) for j in range(grid[1]) for m in range(grid[2])]
    pick = rng.choice(len(cells), size=k, replace=False)
    out, taken = [], set()
    for c in pick:
        i, j, m = cells[int(c)]
        origin = np.array([margin + i * cell, margin + j * cell, margin + m * cell], np.int64)
        pos, dirs = seed(n_subunits, kappa, cell, cell, cell, rng, margin=0, )
        pos = pos - pos.min(axis=0) + origin
        keys = {tuple(p) for p in pos}
        if keys & taken:
            raise RuntimeError("cell placement collided; cell %d is too small for the span" % int(c))
        taken |= keys
        out.append((pos, dirs))
    return out


# ---------------------------------------------------------------------------------------------
def _check():
    """Closed form against a numerical diagonalisation, and the sampler against the closed form."""
    rng = np.random.default_rng(20260909)
    print("closed form against the 12x12 eigensolver")
    print("  kappa        r closed      r numeric     Lp/<b>      modes with amplitude")
    for k in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0):
        B = transfer(k)
        rows = B.sum(axis=1)
        assert np.allclose(rows, rows[0]), "row sums must be equal on one orbit"
        w, v = np.linalg.eigh(B)
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        psi = v[:, 0] * np.sign(v[:, 0].sum())
        amp = np.array([np.sum((psi[:, None] * VEC * v[:, a][:, None]).sum(axis=0) ** 2)
                        for a in range(Z_COORD)])
        nz = amp > 1e-10
        ratios = (w / w[0])[nz]
        r_num = float(ratios[np.argmax(amp[nz])]) if nz.any() else 0.0
        assert abs(rows[0] - z_partition(k)) < 1e-12 * max(1.0, rows[0])
        # One decay ratio, not one eigenvector: at kappa = 0 every non-leading eigenvalue is zero
        # and eigh returns an arbitrary basis of that null space, so the amplitude spreads over
        # more than three of them. The physical claim is that they all decay at the same rate.
        assert np.ptp(ratios) < 1e-10, "every mode carrying amplitude must share one decay ratio"
        assert abs(amp.sum() - 2.0) < 1e-10, "sum rule sum_a A_a = <b^2> = 2"
        assert abs(r_num - ratio(k)) < 1e-10, "closed form must match the eigensolver"
        print("  %5.1f    %12.9f  %12.9f  %10.4f   %d" % (k, ratio(k), r_num, lp_over_b(k), nz.sum()))

    print("\nsampler against the closed form, 200000 draws")
    print("  kappa    M    <R2> sampled   <R2> closed   rel err    self-avoiding")
    for k in (1.0, 2.0, 4.0, 6.0):
        for m in (7, 15, 31, 63):
            r2, _, keep = sample(k, m, 200000, rng)
            exact = mean_square_r(k, m)
            rel = r2.mean() / exact - 1.0
            assert abs(rel) < 0.01, "phantom sampler must reproduce the closed form"
            print("  %5.1f  %3d   %12.3f  %12.3f   %+8.5f   %.4f"
                  % (k, m, r2.mean(), exact, rel, keep.mean()))

    print("\nseeding is unbiased: 3000 seeds against the closed form")
    print("  n  kappa   box   <R2> seeded   <R2> closed   rel err")
    for n, k, box in ((11, 4.0, 32), (11, 1.0, 32), (16, 2.0, 32)):
        r2 = []
        for _ in range(3000):
            pos, dirs = seed(n, k, box, box, box, rng)
            d = pos[-1] - pos[0]
            r2.append(float(d @ d))
            assert len(pos) == n and len(dirs) == n - 1
            assert np.allclose(np.linalg.norm(np.diff(pos, axis=0), axis=1), BOND)
            assert len({tuple(p) for p in pos}) == n, "seed must be self-avoiding"
            assert (pos >= 3).all() and (pos < box - 3).all(), "seed must sit inside the margin"
            assert np.array_equal(G.VECS[dirs], np.diff(pos, axis=0)), "dirs must match the steps"
        # the seed is the SELF-AVOIDING chain, so it sits above the phantom closed form
        exact_saw = sample(k, n - 1, 200000, rng)
        saw = exact_saw[0][exact_saw[2]].mean()
        rel = np.mean(r2) / saw - 1.0
        assert abs(rel) < 0.05, "seeded ensemble must reproduce the exact self-avoiding chain"
        print("  %2d  %4.1f   %3d   %11.3f   %11.3f   %+8.4f  (phantom %.3f)"
              % (n, k, box, np.mean(r2), saw, rel, mean_square_r(k, n - 1)))

    print("\na box too small to hold the ensemble is refused, not silently truncated")
    try:
        seed(65, 6.0, 32, 32, 32, rng)
        raise AssertionError("should have refused a 32 cube for 65 subunits at kappa 6")
    except RuntimeError as exc:
        print("  " + str(exc).split(";")[0])

    print("\nall checks passed")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        _check()
    else:
        ap.print_help()
