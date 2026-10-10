"""Extra (polydisperse) bath species for stamp_msd.py: parsing, counts, pyLM registration, solver diameters, friction.

A spec is `SIGMA_NM:ETA[:D_UM2S]`, repeatable (`--bath-species 5:0.05 --bath-species 8:0.02`). The primary bath (3 nm,
`--eta`, `--d-bath`) stays byte-lattice type 1 and pyLM species 'B'; type 2 is the body's overlay-fill diameter row and is
held by a particle-free placeholder species 'FILL'; extra species become 'B2', 'B3', ... = types 3, 4, ... (pyLM numbers
species in definition order, RDME.py defineSpecies). The default D of an extra species is Stokes-Einstein from the
represented diameters, D_ribo * sigma_ribo / sigma_s, the convention of the primary bath and of the ReaDDy reference.

The engine's friction sum is already polydisperse (struct_friction_dev.cuh: sum over particles of c_t * v_t/V); this module
supplies the per-type coefficients from friction_closure.py, the one place the formula lives.
"""
import numpy as np
import friction_closure as fc

FIRST_EXTRA_TYPE = 3          # byte-lattice particle type of the first extra species ('B2')


class BathSpecies:
    __slots__ = ('sigma', 'eta', 'd_um2', 'name', 'ptype', 'count', 'c')

    def __init__(self, sigma, eta, d_um2, index):
        self.sigma, self.eta, self.d_um2 = float(sigma), float(eta), float(d_um2)
        self.name = f'B{index + 2}'
        self.ptype = FIRST_EXTRA_TYPE + index
        self.count = 0
        self.c = 0.0


def parse(specs, d_ribo_um2, sigma_ribo):
    """specs: list of 'SIGMA:ETA[:D]' strings (or None) -> [BathSpecies]. Raises on a malformed spec."""
    out = []
    for i, sp in enumerate(specs or []):
        parts = sp.split(':')
        if len(parts) not in (2, 3):
            raise ValueError(f"--bath-species expects SIGMA_NM:ETA[:D_UM2S], got '{sp}'")
        sigma, eta = float(parts[0]), float(parts[1])
        if sigma <= 0 or eta < 0:
            raise ValueError(f"--bath-species: sigma must be > 0 and eta >= 0, got '{sp}'")
        d = float(parts[2]) if len(parts) == 3 else d_ribo_um2 * sigma_ribo / sigma
        out.append(BathSpecies(sigma, eta, d, i))
    return out


def count(eta, sigma, nx, ny, nz, nm_per_voxel):
    v_box = nx * ny * nz * nm_per_voxel ** 3
    return int(round(eta * v_box / ((np.pi / 6.0) * sigma ** 3)))


def cfl_check(species, tau_s, nm_per_voxel):
    for s in species:
        q = s.d_um2 * 1e6 * tau_s / nm_per_voxel ** 2
        if 6.0 * q > 0.98:
            raise SystemExit(f"CFL: {s.name} ({s.sigma} nm) 6*D*dt/h^2 = {6*q:.3f} > 0.98")


def define_in_sim(sim, species, nx, ny, nz, nm_per_voxel, region='default'):
    """Define 'FILL' (type 2, no particles) and the extra species in the pyLM simulation, set their D and add particles.
    Call BEFORE the primary bath's addParticles or after; order of defineSpecies is what fixes the types, so this must be
    called right after sim.defineSpecies(['B'])."""
    if not species:
        return
    sim.defineSpecies(['FILL'] + [s.name for s in species])
    for s in species:
        s.count = count(s.eta, s.sigma, nx, ny, nz, nm_per_voxel)
        sim.modifyRegion(region).setDiffusionRate(s.name, s.d_um2 * 1e-12)
        if s.count > 0:
            sim.addParticles(species=s.name, region=region, count=s.count)
        print(f"[bath_species] {s.name} = type {s.ptype}: sigma {s.sigma} nm, eta {s.eta}, D {s.d_um2} um^2/s, "
              f"{s.count} particles ({s.count / (nx * ny * nz):.2f} per voxel)")


def diameters(species, sigma_bath, sigma_fill):
    """The setSpeciesHardSphereDiameters array: index t-1 is type t. Types 1 (primary bath), 2 (fill), 3.. (extra)."""
    n = 2 + len(species)
    d = np.zeros(n, np.float64)
    d[0] = sigma_bath
    d[1] = sigma_fill
    for s in species:
        d[s.ptype - 1] = s.sigma
    return d


def set_couplings(sol, species):
    """No external potential on the extra bath species (gamma = 0)."""
    for s in species:
        sol.setSpeciesCoupling(s.ptype - 1, 0.0, 0.0)


def coefficients(species, sigma_ribo, d_ribo_um2):
    """Free-draining two-body coefficient per extra species (stored on each BathSpecies.c) -> list of (ptype, c)."""
    out = []
    for s in species:
        s.c = fc.coefficient(sigma_ribo, s.sigma, d_ribo_um2, s.d_um2)
        out.append((s.ptype, s.c))
    return out


def predicted_m(c_primary, eta_primary, species):
    return fc.mobility_polydisperse([c_primary] + [s.c for s in species], [eta_primary] + [s.eta for s in species])
