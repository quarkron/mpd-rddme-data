"""Geometry, species table and field for the immune-synapse contact-zone problem.

WHAT THIS IS
    The model of Isaacson & Zhang, Bull. Math. Biol. 87:14 (2024), section IV.D, rebuilt on a
    Cartesian lattice with hard-sphere volume exclusion. Their CRDDME has no two-body
    interactions; they state in that section that they therefore "ignore size-based segregation
    forces", and they attribute the gap between their seconds-scale core filling and the
    physiological tens of minutes to the larger size of the complexes and their repulsive
    interactions with other synapse components. This model puts that exclusion in.

THE ONE NON-OBVIOUS PARAMETER
    The molecules are ANCHORED to a membrane, so their crowding is areal, but the excess free
    energy is a three-dimensional functional that reads a volume fraction out of a cubic voxel.
    Left alone it spreads a jammed monolayer through 40 nm of z and reports a dilute gas. The
    diameter handed to the functional is therefore not the molecule's equivalent-sphere diameter
    but the sphere that carries the same second virial coefficient PER UNIT AREA of the slab:
    a two-dimensional density rho spread over thickness h is a three-dimensional density rho/h,
    so matching the excess free energy per area at leading order gives

        (2/3) pi sigma_eff^3 / h = (1/2) pi sigma^2   =>   sigma_eff = (0.75 h sigma^2)^(1/3).

    This changes a parameter, never a functional. VIRIAL_MATCH=False switches to the packing-
    fraction convention (0.75 -> 1.5), which is the jamming-limit equivalence instead; it raises
    every diameter by 2^(1/3) and leaves every size RATIO unchanged, since both scale as
    sigma^(2/3).

SIZES
    Each species is an elongated ectodomain, not a sphere. `EQUIV` below is the equivalent-sphere
    diameter, computed from the Kihara convex-body excluded volume of a spherocylinder of the
    quoted span and width. That agrees with the Tirado-Garcia de la Torre Stokes-equivalent
    diameter to better than 7% for the three complexes, so the choice does not carry any result.
"""
import math

# --- lattice ---------------------------------------------------------------------------------
H_NM = 40.0                      # voxel edge, nm
R_DISK_UM = 5.6                  # contact-zone radius (their domain)
R_CORE_UM = 2.0                  # actin-depletion region
R_PSMAC_UM = 4.0                 # pSMAC edge; the ramp doubles its slope outside this

# --- field -----------------------------------------------------------------------------------
# phi(r) = f0 r for r <= R_P, and f0 R_P + 2 f0 (r - R_P) beyond. f0 = 1 kT/um is their value,
# taken from Siokis et al. The basis field carries the SHAPE in um and the species coupling
# carries f0 in kT/um, so a sweep in f0 needs no new field.
F0_KT_PER_UM = 1.0

# --- species ---------------------------------------------------------------------------------
# span x width in nm -> equivalent sphere; D in um^2/s; gamma = coupling to the ramp in kT/um.
# Free TCR and pMHC feel no active force and the complex does, exactly as in their model.
SPECIES = [
    # name          span  width   D      couples
    ('TCR',          7.0,  4.0,  0.10,  False),
    ('pMHC',         7.0,  4.0,  0.10,  False),
    ('TCR_pMHC',    15.0,  5.0,  0.06,  True),
    ('LFA_ICAM',    42.0,  5.0,  0.06,  False),
    ('CD45',        22.0,  4.0,  0.10,  False),
]
MINIMAL = ('TCR', 'pMHC', 'TCR_pMHC')          # the three that reproduce their figure

# --- gauge -----------------------------------------------------------------------------------
# MEASURED, 2026-09-07: a one-body potential cannot act on transport alone in this scheme. Every
# acceptance takes Delta U, reactions included (the Frohner-Noe factor exp(-Delta U) in the
# reaction kernel), so coupling only the complex to the ramp makes the binding equilibrium
# position dependent, K(r) = K_id exp(-beta Phi_C(r)). Across their domain Phi_C spans 7.2 kT, a
# factor of 1340 in K. The field-free well-mixed gate reproduces the analytic kinetics to 1%; the
# same run with f0 = 1 kT/um reproduces 6.9% of it, against 6.0% predicted by the area-weighted
# mean of exp(-psi). This is not a switch that can be turned off, and it applies to any scheme
# that claims detailed balance of reaction fluxes under a one-body potential.
#
#   'db'      Phi = 0 on the free species and f0 psi on the complex. Their species choice,
#             detailed-balance consistent, and the binding equilibrium varies with radius.
#   'neutral' Phi = f0 psi / 2 on each free species and f0 psi on the complex, so Delta U of the
#             reaction is identically zero and K is uniform. The complex still drifts with the
#             full force. This isolates transport and exclusion from the chemical artifact, at
#             the cost of giving the free species half the drift their model gives them none of.
GAUGES = ('db', 'neutral')


def coupling_weight(name, gauge):
    """Multiplier on f0 for each species, per gauge. See the GAUGES note."""
    if name == 'TCR_pMHC':
        return 1.0
    if name in ('TCR', 'pMHC'):
        return 0.5 if gauge == 'neutral' else 0.0
    return 0.0                                  # LFA-1-ICAM-1 and CD45 feel no active force

# --- chemistry -------------------------------------------------------------------------------
# Siokis et al. Cell Rep. 24 (2018), the source both their model and ours draw on.
K_ON_TCR = 2.0e4      # M^-1 s^-1
K_OFF_TCR = 0.1       # s^-1   (their mu)
K_ON_LFA = 3.0e5      # M^-1 s^-1
K_OFF_LFA = 0.03      # s^-1
RXN_RADIUS_NM = 15.0  # their epsilon, the mean length of a TCR-pMHC complex

# --- initial condition -----------------------------------------------------------------------
# They seed 10 free TCR and 10 pMHC per mesh polygon outside the actin-depletion region. Their
# mesh is 13890 polygons over a disk of radius 5.6 um, so a polygon is 7093 nm^2 and the surface
# density is 1410 per um^2. That is roughly fourteen times the physiological TCR density and
# sixty times the density Siokis uses, which is why it is a parameter here and not a constant:
# the whole question of whether exclusion is decisive or marginal turns on it.
SIGMA_SURF_THEIRS = 1410.0     # per um^2, their initialization
SIGMA_SURF_PHYS = 100.0        # per um^2, physiological TCR
SIGMA_SURF_SIOKIS = 9.0        # per um^2, Siokis TCR


def kihara_equiv_diameter(span_nm, width_nm):
    """Equivalent hard-sphere diameter of a spherocylinder, matching the orientationally
    averaged pair excluded volume. Kihara's convex-body result V_excl = 2V + 2RS, with R the
    mean radius of curvature. For a sphere this returns the sphere's own diameter."""
    a = 0.5 * width_nm
    L = max(span_nm - width_nm, 0.0)             # cylindrical part
    V = math.pi * a * a * L + (4.0 / 3.0) * math.pi * a ** 3
    S = 2.0 * math.pi * a * L + 4.0 * math.pi * a * a
    R = a + 0.25 * L
    v_excl = 2.0 * V + 2.0 * R * S
    return (3.0 * v_excl / (4.0 * math.pi)) ** (1.0 / 3.0)


def slab_equivalent_diameter(sigma_nm, h_nm=H_NM, virial_match=True):
    """Diameter to hand the three-dimensional functional so that a one-voxel-thick layer of
    ANCHORED molecules carries the right excess free energy per unit area. See the module
    docstring: virial matching uses 0.75, packing-fraction matching uses 1.5."""
    c = 0.75 if virial_match else 1.5
    return (c * h_nm * sigma_nm ** 2) ** (1.0 / 3.0)


def species_table(names=None, h_nm=H_NM, virial_match=True):
    """(name, D_um2_s, couples, sigma_equiv_nm, sigma_lattice_nm) for the requested species."""
    want = list(names) if names is not None else [s[0] for s in SPECIES]
    out = []
    for name, span, width, D, couples in SPECIES:
        if name not in want:
            continue
        eq = kihara_equiv_diameter(span, width)
        out.append((name, D, couples, eq, slab_equivalent_diameter(eq, h_nm, virial_match)))
    return out


def ramp_shape_um(r_um):
    """The Isaacson-Zhang potential divided by f0, in um. Continuous at R_PSMAC by construction:
    the outer branch is anchored to the inner branch's value there, not restarted from zero."""
    inner = r_um
    outer = R_PSMAC_UM + 2.0 * (r_um - R_PSMAC_UM)
    return inner if r_um <= R_PSMAC_UM else outer


def core_annulus_areas_um2():
    """Areas of the core and of the seeded annulus outside it."""
    a_core = math.pi * R_CORE_UM ** 2
    a_disk = math.pi * R_DISK_UM ** 2
    return a_core, a_disk - a_core


if __name__ == '__main__':
    a_core, a_ann = core_annulus_areas_um2()
    print('voxel %.0f nm   disk R %.1f um   core R %.1f um   pSMAC R %.1f um'
          % (H_NM, R_DISK_UM, R_CORE_UM, R_PSMAC_UM))
    print('core area %.2f um^2   seeded annulus %.2f um^2   ratio %.2f'
          % (a_core, a_ann, a_ann / a_core))
    print('%-10s %6s %6s %8s %10s %10s' % ('species', 'span', 'width', 'D', 'equiv', 'lattice'))
    for name, span, width, D, _c in SPECIES:
        eq = kihara_equiv_diameter(span, width)
        print('%-10s %6.1f %6.1f %8.2f %10.2f %10.2f'
              % (name, span, width, D, eq, slab_equivalent_diameter(eq)))
    print('phi(0)=%.2f  phi(R_C)=%.2f  phi(R_P)=%.2f  phi(R)=%.2f  kT at f0=%.1f kT/um'
          % tuple([F0_KT_PER_UM * ramp_shape_um(r)
                   for r in (0.0, R_CORE_UM, R_PSMAC_UM, R_DISK_UM)] + [F0_KT_PER_UM]))
    for label, dens in (('theirs', SIGMA_SURF_THEIRS), ('physiological', SIGMA_SURF_PHYS),
                        ('Siokis', SIGMA_SURF_SIOKIS)):
        n = dens * a_ann
        print('  %-14s %8.0f per um^2 -> %9.0f molecules seeded' % (label, dens, n))
