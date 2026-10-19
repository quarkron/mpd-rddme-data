"""Shared definition of the three reaction schemes of the advantage-metric campaign.

ONE SOURCE OF TRUTH for both engines. adv_lattice.py and adv_readdy.py both import this, so a
species diameter or a rate constant cannot drift between the arms; that drift is exactly what
would turn a fair comparison into a meaningless one.

The three schemes, all at h = 10 nm, D = 1e5 nm^2/s, in a 32 x 8 x 8 subvolume box:

  diff  A, B                      no reactions. The transport ceiling alone.
  rev   A + B <=> C               the published reversible bimolecular test. sigma_C conserves
                                  excluded volume in the dilute limit.
  net   A + B <=> C               a synthetic mass-action network. Three reversible pairs, six
        C     <=> D               channels, five species. The species graph is a TREE, so the
        D + A <=> E               stoichiometric matrix has no cycle and the Wegscheider
                                  conditions are satisfied for ANY choice of the six rates. A
                                  cyclic network would need its rates constrained to be
                                  thermodynamically consistent, and an inconsistent one has no
                                  equilibrium for the detailed-balance handler to reach.

RATE SCALE. k_off is deliberately 200 /s, not the 5e5 /s of the published timing configuration.
At 5e5 the lattice reaction operator is past its own single-firing ceiling, a_v tau = 47 at the
nominal step, so a timing measurement there is not a measurement of the method. At 200 /s the
transport ceiling binds and the advantage is measured in the regime the method is valid in. The
reaction ceiling at any OTHER k_off is then recovered analytically from the same occupancy field,
since per-step cost does not depend on the rate; see adv_analyze.reaction_ceiling. That is why the
campaign does not sweep k_off in wall time.

The equilibrium constants are per SYSTEM volume for the lattice arm, K_id = (c_on/c_off)*V_box.
Using V_sub is wrong by the subvolume count. See lmrddme-reactive-doi-rxmc.
"""
import math

H_NM = 10.0                      # subvolume edge
V_SUB = H_NM ** 3                # nm^3
D_NM2S = 1.0e5                   # every species' base diffusion coefficient, nm^2/s
DT_S = 50e-6                     # nominal lattice timestep
DIMS = (32, 8, 8)                # 320 x 80 x 80 nm, the published benchmark box
M_SUB = DIMS[0] * DIMS[1] * DIMS[2]
V_BOX = M_SUB * V_SUB

K_SOFT = 50.0                    # ReaDDy harmonic stiffness, kT/nm^2, as in every other arm
K_EQ = 200.0                     # binding constant of every bimolecular pair, nm^3
K_OFF = 200.0                    # dissociation / reverse rate of every channel, 1/s
R_DOI = 4.5                      # ReaDDy reaction radius, nm

# Sinusoidal one-body field, identical to the published reacting configuration.
GAMMA = 0.3                      # kT
WAVELENGTH = 80.0                # nm = 8h


def _vol_conserving(*sigmas):
    """Diameter of a product that carries the summed excluded volume of its reactants."""
    return (sum(s ** 3 for s in sigmas)) ** (1.0 / 3.0)


SIG_A = 3.0
SIG_B = 3.0
SIG_C = _vol_conserving(SIG_A, SIG_B)          # 3.7798
SIG_D = SIG_C                                  # isomer of C, same volume
SIG_E = _vol_conserving(SIG_D, SIG_A)          # 4.3267


SCHEMES = {
    'diff': dict(
        species=['A', 'B'],
        sigma={'A': SIG_A, 'B': SIG_B},
        # (kind, reactants, products, rate). kind is 'fusion', 'fission' or 'conversion'.
        reactions=[],
        seeded=['A', 'B'],
    ),
    'rev': dict(
        species=['A', 'B', 'C'],
        sigma={'A': SIG_A, 'B': SIG_B, 'C': SIG_C},
        reactions=[
            ('fusion',   ('A', 'B'), ('C',),      None),     # rate filled in by rates()
            ('fission',  ('C',),     ('A', 'B'),  K_OFF),
        ],
        seeded=['A', 'B'],
    ),
    'net': dict(
        species=['A', 'B', 'C', 'D', 'E'],
        sigma={'A': SIG_A, 'B': SIG_B, 'C': SIG_C, 'D': SIG_D, 'E': SIG_E},
        reactions=[
            ('fusion',     ('A', 'B'), ('C',),     None),
            ('fission',    ('C',),     ('A', 'B'), K_OFF),
            ('conversion', ('C',),     ('D',),     K_OFF),
            ('conversion', ('D',),     ('C',),     K_OFF),
            ('fusion',     ('D', 'A'), ('E',),     None),
            ('fission',    ('E',),     ('D', 'A'), K_OFF),
        ],
        seeded=['A', 'B'],
    ),
}


TERNARY = dict(
    species=['A', 'B', 'C'],
    sigma={'A': 2.0, 'B': 4.0, 'C': 20.0},
    reactions=[],
    seeded=['A', 'B', 'C'],
    eta_a=0.07, eta_b=0.07, c_per_nm3=1.0 / 80.0 ** 3,
)
SCHEMES['ternary'] = TERNARY

# ---- The reaction-count sweep of the supplement figure (added 2026-09-30) --------------------
# R independent reversible pairs A_i + B_i <=> C_i at ONE packing fraction, with the same seeded
# particles split evenly over the R pairs and every diameter that of `rev`. So the particle count,
# the sizes and the packing fraction are identical at every R and only the number of reaction
# channels (2R) and species (3R) changes. Each pair's binding constant is R*K_EQ: with 1/R of the
# particles per pair, that keeps each pair's bound fraction, and the total event rate, equal to the
# single-pair scheme, so the x-axis measures the cost of carrying more channels, not of firing more
# events. pr0 is diffusion only (A, B), pr1 is `rev` under other species names.
PAIR_COUNTS = (0, 1, 2, 3, 4, 6)
ETA_PAIRS = 0.09                 # the sinusoidal system of the main text (Table I)


def _pairs_scheme(r):
    if r == 0:
        return dict(species=['A', 'B'], sigma={'A': SIG_A, 'B': SIG_B}, reactions=[],
                    seeded=['A', 'B'], n_pairs=0)
    sp, sg, rx, seeded = [], {}, [], []
    for i in range(1, r + 1):
        a, b, c = 'A%d' % i, 'B%d' % i, 'C%d' % i
        sp += [a, b, c]
        sg.update({a: SIG_A, b: SIG_B, c: SIG_C})
        rx += [('fusion', (a, b), (c,), None), ('fission', (c,), (a, b), K_OFF)]
        seeded += [a, b]
    return dict(species=sp, sigma=sg, reactions=rx, seeded=seeded, n_pairs=r, k_eq=r * K_EQ)


for _r in PAIR_COUNTS:
    SCHEMES['pr%d' % _r] = _pairs_scheme(_r)


def ternary_counts():
    """Counts of the main-text ternary scene at DIMS: A and B at eta 0.07 each, and oversize C at
    one per (80 nm)^3, which is the published number density and what the particle arm is given."""
    sig = TERNARY['sigma']
    return {
        'A': int(round(TERNARY['eta_a'] * V_BOX / (math.pi / 6 * sig['A'] ** 3))),
        'B': int(round(TERNARY['eta_b'] * V_BOX / (math.pi / 6 * sig['B'] ** 3))),
        'C': int(round(TERNARY['c_per_nm3'] * V_BOX)),
    }


def diffusion(scheme):
    """Stokes-scaled diffusion coefficients, nm^2/s. The smallest species carries D_NM2S."""
    sig = SCHEMES[scheme]['sigma']
    smin = min(sig.values())
    return {s: D_NM2S * smin / d for s, d in sig.items()}


def rates(scheme, convention='pylm'):
    """Resolve every reaction rate in the requested convention.

    ★THE TRAP THIS FUNCTION EXISTS FOR. A bimolecular rate has two conventions in this codebase
    and they differ by the subvolume count M, here 2048:

      'pylm'   c_on = K_EQ*K_OFF / V_BOX   what pyLM's addReaction takes; LM divides by M
                                           internally to reach the per-subvolume propensity.
      'kernel' k_f  = K_EQ*K_OFF / V_SUB   what the propensity k_f*n_A*n_B is actually built
                                           from, and therefore what adv_cpu's --rxn takes.

    Passing the pylm value to the kernel convention under-drives the reaction by 2048 and the
    run looks almost non-reacting, which is exactly how this was caught. First-order rates are
    the same in both. Returns a list parallel to SCHEMES[scheme]['reactions'].
    """
    if convention not in ('pylm', 'kernel'):
        raise ValueError('convention must be pylm or kernel, got %r' % (convention,))
    vol = V_BOX if convention == 'pylm' else V_SUB
    k_eq = SCHEMES[scheme].get('k_eq', K_EQ)
    out = []
    for kind, react, prod, rate in SCHEMES[scheme]['reactions']:
        if rate is None:
            rate = k_eq * K_OFF / vol
        out.append((kind, react, prod, rate))
    return out


def doi_lambda(scheme=None):
    """ReaDDy microscopic association rate that reproduces K_EQ over the reaction ball.

    The accessible volume is the ball minus the contact core, not the full ball; using the full
    ball overstates the constant by ~30% at R = 1.5 sigma. See lmrddme-reactive-doi-rxmc.
    """
    sig_ab = 0.5 * (SIG_A + SIG_B)
    k_eq = SCHEMES[scheme].get('k_eq', K_EQ) if scheme else K_EQ
    return k_eq * K_OFF / ((4.0 / 3.0) * math.pi * (R_DOI ** 3 - sig_ab ** 3))


def bh_contact(target, k=K_SOFT, n=4000):
    """Contact distance whose Barker-Henderson effective diameter equals `target`.

    Trapezoid on d_eff(d) = int_0^d [1 - exp(-beta k (r-d)^2 / 2)] dr, bisected. Identical to the
    function every other ReaDDy arm of this project uses; duplicated here rather than imported so
    the campaign has no cross-folder import at run time.
    """
    def d_eff(d):
        h = d / n
        tot = 0.0
        for i in range(n + 1):
            r = i * h
            w = 0.5 if (i == 0 or i == n) else 1.0
            tot += w * (1.0 - math.exp(-0.5 * k * (r - d) ** 2))
        return tot * h
    lo, hi = target, target + 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if d_eff(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def counts_for_eta(scheme, eta):
    """Initial counts of the seeded species giving total packing fraction `eta` once the
    reaction has equilibrated.

    The seeded species are the ones present at t = 0; products form from them. Because every
    reaction in every scheme conserves excluded volume by construction of the diameters, the
    packing fraction is invariant under reaction and is fixed by the seeded counts alone. That is
    the property that lets one crowding axis serve all three schemes.
    """
    if scheme == 'ternary':
        return ternary_counts()
    sig = SCHEMES[scheme]['sigma']
    seeded = SCHEMES[scheme]['seeded']
    vol_per_set = sum((math.pi / 6.0) * sig[s] ** 3 for s in seeded)
    n_sets = eta * V_BOX / vol_per_set
    if SCHEMES[scheme].get('n_pairs', 0) > 1:
        # One pair's worth of seeded volume is split over the R pairs: the same particle count as
        # R = 1, with the remainder of the division spread over the first pairs.
        r = SCHEMES[scheme]['n_pairs']
        per = eta * V_BOX / (vol_per_set / r)       # A+B sets in total, as for one pair
        tot = int(round(per))
        base, extra = divmod(tot, r)
        out = {}
        for i in range(r):
            n = base + (1 if i < extra else 0)
            out['A%d' % (i + 1)] = n
            out['B%d' % (i + 1)] = n
        return out
    return {s: int(round(n_sets)) for s in seeded}


def eta_of_counts(scheme, counts):
    sig = SCHEMES[scheme]['sigma']
    return (math.pi / 6.0) * sum(counts.get(s, 0) * sig[s] ** 3 for s in sig) / V_BOX


def equilibrium_per_voxel(scheme, eta):
    """Mass-action equilibrium composition per subvolume at packing fraction `eta`, ideal gas.

    Used to size initial conditions and to sanity-check a finished run, NOT to replace the
    measured occupancy field that the ceilings are computed from. Concentrations in nm^-3.
    """
    c0 = counts_for_eta(scheme, eta)
    a_tot = c0.get('A', 0) / V_BOX
    b_tot = c0.get('B', 0) / V_BOX
    if scheme in ('diff', 'ternary'):
        c = counts_for_eta(scheme, eta)
        return {k: v / M_SUB for k, v in c.items()}
    K1 = K_EQ
    if scheme == 'rev':
        # a + c = a_tot, b + c = b_tot, c = K1 a b
        a = b = None
        lo, hi = 0.0, min(a_tot, b_tot)
        for _ in range(200):
            c = 0.5 * (lo + hi)
            if K1 * (a_tot - c) * (b_tot - c) > c:
                lo = c
            else:
                hi = c
        c = 0.5 * (lo + hi)
        return {'A': (a_tot - c) * V_SUB, 'B': (b_tot - c) * V_SUB, 'C': c * V_SUB}
    # net: c = K1 a b, d = c (K2 = 1 by equal forward and reverse conversion rates), e = K3 d a
    K3 = K_EQ
    lo, hi = 0.0, a_tot
    for _ in range(300):
        a = 0.5 * (lo + hi)
        # solve b from the B balance at this a: b + c + d + e = b_tot with c = d = K1 a b,
        # e = K3 K1 a^2 b  ->  b (1 + 2 K1 a + K3 K1 a^2) = b_tot
        b = b_tot / (1.0 + 2.0 * K1 * a + K3 * K1 * a * a)
        c = K1 * a * b
        d = c
        e = K3 * d * a
        if a + c + d + 2.0 * e > a_tot:
            hi = a
        else:
            lo = a
    a = 0.5 * (lo + hi)
    b = b_tot / (1.0 + 2.0 * K1 * a + K3 * K1 * a * a)
    c = K1 * a * b
    d = c
    e = K3 * d * a
    return {k: v * V_SUB for k, v in (('A', a), ('B', b), ('C', c), ('D', d), ('E', e))}


ETA_SWEEP = (0.02, 0.04, 0.06, 0.09, 0.12, 0.15)


if __name__ == '__main__':
    print('box %s = %d subvolumes = %.0f nm^3, V_sub %.0f nm^3'
          % (DIMS, M_SUB, V_BOX, V_SUB))
    print('diameters: A %.3f B %.3f C %.4f D %.4f E %.4f nm'
          % (SIG_A, SIG_B, SIG_C, SIG_D, SIG_E))
    print('Doi lambda %.4g /s ; BH contacts: AA %.4f AC %.4f nm'
          % (doi_lambda(), bh_contact(SIG_A), bh_contact(0.5 * (SIG_A + SIG_C))))
    print('fusion rate: pylm %.6g /s, kernel %.6g /s (ratio %d = M)'
          % (rates('rev', 'pylm')[0][3], rates('rev', 'kernel')[0][3], M_SUB))
    for sc in ('diff', 'rev', 'net'):
        print('\n== %s: species %s' % (sc, SCHEMES[sc]['species']))
        for eta in ETA_SWEEP:
            c0 = counts_for_eta(sc, eta)
            eq = equilibrium_per_voxel(sc, eta)
            tot = sum(eq.values())
            print('  eta %.3f  seed %s  -> per voxel %s  total %.2f'
                  % (eta, {k: v for k, v in c0.items()},
                     {k: round(v, 3) for k, v in eq.items()}, tot))
