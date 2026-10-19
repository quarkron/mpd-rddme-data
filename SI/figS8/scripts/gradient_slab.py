"""Demixed (nonequilibrium) starts for gradient_growth.py (2026-10-08): crowders and monomers begin in separate regions.

--start slab     crowders in x < L/2, monomers in x >= L/2 (one flat interface)
--start stripes  four alternating layers along x, 160 nm each at L = 640 nm (crowder, monomer, crowder, monomer)
--start droplet  monomers in a central sphere holding half the box volume, crowders outside it
--start cubes    3D checkerboard of 160 nm cubes, crowder and monomer alternating
Each region is half the box, so every layout starts at the same local densities. The small bath, if any, stays uniform;
the well is flat when phi_core == phi_edge, so afterwards only the components' own interactions act. With volume
exclusion the separated phases may persist; without it (point particles) they should mix. Used for the one-tone
block d experiment of fig:polymer.
"""
import numpy as np

KINDS = ('slab', 'stripes', 'droplet', 'cubes')


def crowder_mask(nx, kind):
    """Boolean per voxel, flattened like psi (v = z*nx*nx + y*nx + x): True where crowders start."""
    idx = np.arange(nx)
    z, y, x = np.meshgrid(idx, idx, idx, indexing='ij')
    if kind == 'slab':
        m = x < nx // 2
    elif kind == 'stripes':
        m = (x // (nx // 4)) % 2 == 0
    elif kind == 'droplet':
        c = (nx - 1) / 2.0
        r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
        m = r > np.quantile(r, 0.5)                                   # monomer sphere = the inner half of the volume
    elif kind == 'cubes':
        b = nx // 4
        m = ((x // b) + (y // b) + (z // b)) % 2 == 0
    else:
        raise ValueError(kind)
    return m.reshape(-1)


def crowder_half(nx, kind='slab'):
    """Placement weight for the crowders: 1 in the crowder region, 0 elsewhere."""
    return crowder_mask(nx, kind).astype(np.float64)


def add_monomers(sim, n, nx, rng, kind='slab', species='M'):
    """n monomers placed uniformly over the monomer region (the complement of the crowder region)."""
    sim.getLattice()
    free = np.flatnonzero(~crowder_mask(nx, kind))
    for v in rng.choice(free, size=n):
        z, rem = divmod(int(v), nx * nx); y, x = divmod(rem, nx)
        sim.addParticleAtIdx((int(x), int(y), int(z)), species)


def place_bodies(n, nx, rng, kind='slab'):
    """n body centres in the crowder region, no two within 2 voxels on any axis (20 nm spheres at most at contact).

    ★Random sequential placement cannot do this: confined to half the box, the published 12 516 bodies cover 76 % of
    the region's 2x2x2 slots, and random placement on this lattice jams near 40 %. They are drawn instead as a random
    subset of the even sublattice (x, y, z all even) inside the region: overlap-free by construction, a dense packing
    that melts as soon as the run starts."""
    m = crowder_mask(nx, kind).reshape(nx, nx, nx)                     # [z, y, x]
    z, y, x = np.nonzero(m)
    ok = (x % 2 == 0) & (y % 2 == 0) & (z % 2 == 0)
    sites = np.stack([x[ok], y[ok], z[ok]], 1)
    if n > len(sites):
        raise RuntimeError(f'{n} bodies but only {len(sites)} sublattice sites in the {kind} crowder region')
    pick = rng.choice(len(sites), size=n, replace=False)
    return [tuple(int(c) for c in sites[i]) for i in pick]
