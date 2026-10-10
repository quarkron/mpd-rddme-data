"""Two-body collisional friction closure for a large tracer in a bath of small Brownian spheres (free-draining).

Derivation: LM-RDDME manuscript repo, tex/oversize_02_collisional_friction.tex. Long-time self-diffusion of a
tracer T among bath spheres B without hydrodynamic interactions, to first order in the bath volume fraction:

    D_L / D_T = 1 - c * phi_B,     c = (1/2) * D_T / (D_T + D_B) * rho_B * (4 pi / 3) a^3 / phi_B
                                    = (1/2) * D_T / (D_T + D_B) * (1 + lam)^3 / lam^3,     lam = sigma_B / sigma_T,

resummed as an additive friction (Medina-Noyola 1988 structure) to m(phi) = 1 / (1 + c phi). With Stokes-Einstein
bare coefficients, D ~ 1/sigma, c = (1 + lam)^2 / (2 lam^2); for equal spheres c = 2, the textbook 1 - 2 phi.
The engine takes c per byte-lattice particle type (setStructureFrictionCoefficient) and applies m at the shell
around each body; this module is the ONLY place the formula lives (the tests import it, the driver imports it).

The coefficient is a modelling input. This is the FREE-DRAINING value, the right one for comparison with
Brownian dynamics without hydrodynamics (ReaDDy). A hydrodynamic bath would call for the Einstein-Batchelor
effective viscosity instead (tex/oversize_04_hydrodynamics_reference.tex); see `coefficient_hydrodynamic`.
"""
import math
import sys


def coefficient(sigma_t, sigma_b, d_t, d_b):
    """Free-draining two-body coefficient c for tracer diameter sigma_t, bath diameter sigma_b, bare D_t, D_b.
    Any consistent units (the result is dimensionless)."""
    if sigma_t <= 0 or sigma_b <= 0 or d_t <= 0 or d_b <= 0:
        raise ValueError("diameters and diffusion coefficients must be positive")
    lam = sigma_b / sigma_t
    return 0.5 * (d_t / (d_t + d_b)) * (1.0 + lam) ** 3 / lam ** 3


def coefficient_stokes_einstein(lam):
    """c for D ~ 1/sigma: (1 + lam)^2 / (2 lam^2), lam = sigma_b / sigma_t."""
    if lam <= 0:
        raise ValueError("lam must be positive")
    return (1.0 + lam) ** 2 / (2.0 * lam ** 2)


def coefficient_collective(sigma_t, sigma_b, d_t, d_b, phi):
    """Variant with the bath's collective diffusion D_b (1 + 8 phi) in place of the bare D_b (note 2's heuristic:
    the pile-up ahead of the tracer relaxes at the collective rate). Printed for the record, not used by default."""
    return coefficient(sigma_t, sigma_b, d_t, d_b * (1.0 + 8.0 * phi))


def coefficient_hydrodynamic(phi):
    """Einstein-Batchelor effective-viscosity form, size-ratio independent for sigma_t >> sigma_b:
    sum_s c_s phi_s -> 2.5 phi + 6.2 phi^2, returned as an effective c so that m = 1/(1 + c phi)."""
    if phi <= 0:
        return 2.5
    return 2.5 + 6.2 * phi


def mobility(eta, c):
    """m = 1 / (1 + c eta)."""
    return 1.0 / (1.0 + c * eta)


def coefficients(sigma_t, d_t, species):
    """Polydisperse bath: species = [(sigma_s, d_s), ...] -> [c_s, ...], one free-draining two-body coefficient per bath
    species. The frictions add (note 2, polydisperse generalisation), so the engine sums c_s * phi_s over the shell."""
    return [coefficient(sigma_t, s_s, d_t, d_s) for s_s, d_s in species]


def mobility_polydisperse(cs, phis):
    """m = 1 / (1 + sum_s c_s phi_s) for matching lists of coefficients and packing fractions."""
    if len(cs) != len(phis):
        raise ValueError("cs and phis must have the same length")
    return 1.0 / (1.0 + sum(c * p for c, p in zip(cs, phis)))


def _selfcheck():
    c_equal = coefficient(3.0, 3.0, 1.0, 1.0)
    ok = abs(c_equal - 2.0) < 1e-12 and abs(coefficient_stokes_einstein(1.0) - 2.0) < 1e-12
    print(f"equal spheres: c = {c_equal:.4f} (expected 2.0000)  {'OK' if ok else 'BAD'}")
    # Benchmark of testing/engine/stamp_msd.py: 20 nm body, 3 nm bath, D_bath = D_ribo * 20/3.
    st, sb, dt = 20.0, 3.0, 0.001
    c = coefficient(st, sb, dt, dt * st / sb)
    print(f"benchmark 20/3 nm: c = {c:.2f} (Stokes-Einstein form {coefficient_stokes_einstein(sb / st):.2f})")
    for eta in (0.05, 0.10, 0.15):
        print(f"  eta {eta:.2f}: m = {mobility(eta, c):.3f}   collective-D_b variant m = "
              f"{mobility(eta, coefficient_collective(st, sb, dt, dt * st / sb, eta)):.3f}   "
              f"hydrodynamic m = {mobility(eta, coefficient_hydrodynamic(eta)):.3f}")
    # Polydisperse example used by test_friction_polydisperse.py: 3 nm at eta 0.10 plus 5 nm at eta 0.05, D ~ 1/sigma.
    cs = coefficients(st, sb, [(3.0, dt * st / 3.0), (5.0, dt * st / 5.0)]) if False else \
         coefficients(st, dt, [(3.0, dt * st / 3.0), (5.0, dt * st / 5.0)])
    print(f"polydisperse 3 nm @ 0.10 + 5 nm @ 0.05: c = {cs[0]:.2f}, {cs[1]:.2f}; m = {mobility_polydisperse(cs, [0.10, 0.05]):.3f}"
          f" (5 nm alone at 0.05: m = {mobility(0.05, cs[1]):.3f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selfcheck())
