"""Shared construction of the polymer-crowding demixing scene for every ReaDDy arm.

ONE SOURCE OF TRUTH, for the same reason adv_schemes.py exists. The timing arm and the transport
arm must price and measure the SAME system: if a diameter, a stiffness or a reaction radius drifted
between them, the cost ratio and the mobility validation would describe different physics and
neither could be quoted beside the other.

The scene reproduces the lattice demixing run particle for particle: a 3 nm bath at volume fraction
`bath_eta`, 20 nm crowders at `phi_crowd`, and a semi-dilute polymer of 10 nm beads in chains, in a
cubic periodic box of `nx` lattice voxels at h = 10 nm.

Two deliberate choices, carried over from the timing arm:

  * The crowder-crowder repulsion is KEPT. The lattice arm excludes body from body, so dropping it
    would be different physics. It also sets the neighbour cutoff to 20 nm for the whole system,
    which is the largest single cost in the run; that cost is real and belongs in the ratio.
  * Chains carry the bond length of the LATTICE chain, sqrt(2) h, not the bead diameter. The FCC
    edge orbit the lattice chain lives on has rest length sqrt(2) in voxels, so a ReaDDy chain bonded
    at 10 nm would be a denser object than the one being validated against.

Chains are seeded at the length the lattice scene reaches rather than nucleated from free monomer:
ReaDDy's spatial topology reactions need an existing topology on one side, so a topology cannot form
from two free particles.

`--static-chains` removes the growth and shrink channels and leaves the bonded chains in place. The
transport arm uses it: with the channels live, a chain bead changes identity mid-trajectory and a
mean-square displacement over a fixed particle set is not defined. The crowders never react in
either case, so the crowder observable is unaffected by the choice.
"""
import math
import os

import numpy as np
import readdy

import adv_schemes as S

H = 10.0                      # lattice voxel edge the scene is matched to, nm
BOND_NM = math.sqrt(2.0) * H  # FCC edge orbit: the lattice chain's rest length


def add_args(ap):
    """The flags shared by every driver of this scene. A driver adds its own on top."""
    ap.add_argument('--nx', type=int, default=64, help='box edge in lattice voxels (h = 10 nm)')
    ap.add_argument('--phi-crowd', dest='phi_crowd', type=float, default=0.20)
    ap.add_argument('--bath-eta', dest='bath_eta', type=float, default=0.05)
    ap.add_argument('--n-monomers', dest='n_monomers', type=int, default=60000)
    ap.add_argument('--mean-len', dest='mean_len', type=int, default=19,
                    help='mean chain length the lattice scene reaches; chains are seeded at it')
    ap.add_argument('--n-chains', dest='n_chains', type=int, default=None,
                    help='number of chains to seed. Default puts EVERY monomer in a chain; the '
                         'lattice scene at steady state leaves a large free-monomer pool, and '
                         'matching it means seeding fewer chains than that and letting the rest '
                         'stay free.')
    ap.add_argument('--state-from', dest='state_from', default=None,
                    help='a lattice demixing npz (gradient_growth.py): seed its FINAL chains bead for bead and '
                         'its last crowder snapshot, instead of straight strings of --mean-len at random')
    ap.add_argument('--sigma-bath', dest='sigma_bath', type=float, default=3.0)
    ap.add_argument('--sigma-bead', dest='sigma_bead', type=float, default=10.0)
    ap.add_argument('--sigma-crowd', dest='sigma_crowd', type=float, default=20.0)
    ap.add_argument('--d-bath', dest='d_bath', type=float, default=1.5e5,
                    help='nm^2/s; the lattice scene runs the bath at 0.15 um^2/s')
    ap.add_argument('--d-mono', dest='d_mono', type=float, default=3.0e5,
                    help='nm^2/s; the lattice scene runs monomers and chain beads at 0.3 um^2/s')
    ap.add_argument('--d-crowd', dest='d_crowd', type=float, default=4.0e4,
                    help='nm^2/s; the lattice scene runs the 20 nm bodies at 0.04 um^2/s')
    ap.add_argument('--k-poly', dest='k_poly', type=float, default=20.0)
    ap.add_argument('--k-depoly', dest='k_depoly', type=float, default=3.0)
    ap.add_argument('--no-crowder-pair', dest='no_crowder_pair', action='store_true',
                    help='drop the crowder-crowder repulsion (different physics from the lattice)')
    ap.add_argument('--no-chains', dest='no_chains', action='store_true', help='bath and crowders only')
    ap.add_argument('--no-bath', dest='no_bath', action='store_true',
                    help='drop the 3 nm bath (a reference arm, not the demixing scene)')
    ap.add_argument('--static-chains', dest='static_chains', action='store_true',
                    help='bonded chains with no growth or shrink channel; required for a trajectory')
    ap.add_argument('--arm', default='sc', choices=('sc', 'cpu'))
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--dt', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    return ap


def resolve(A):
    """Everything the scene is determined by, computed once so both drivers report the same."""
    L = A.nx * H
    V = L ** 3
    sig = {'B': A.sigma_bath, 'M': A.sigma_bead, 'C': A.sigma_crowd}
    # EVERY coefficient is the lattice scene's own, not a Stokes chain built from one of them. The
    # lattice run deliberately does NOT obey Stokes across its species: the 3 nm bath is slowed to
    # 0.15 um^2/s to keep the collective hop under the byte lattice's CFL bound while the 10 nm
    # monomer keeps 0.3 um^2/s. Rebuilding the beads' coefficient from the bath by a diameter ratio
    # gave 4.5e4 nm^2/s, 6.7x too slow, which is a different obstacle field for the crowder to push
    # through and a different stability step for the engine.
    D = {'B': A.d_bath, 'M': A.d_mono, 'C': A.d_crowd}
    # The Euler-Maruyama stability step is set by the FASTEST species against the softest contact,
    # tau_R = 2/(D k); quoting it off the campaign's nominal D would understate it for this scene.
    tau_stab = 2.0 / (max(D.values()) * S.K_SOFT)
    # A bonded bead is held by two bonds of the same stiffness, and a harmonic chain's stiffest normal
    # mode has eigenvalue 4k, so with chains present the explicit step must also satisfy 2/(4 k D_M).
    # The pair bound alone put the step 1.6x past this and the chains blew up (bonds 14 -> 48 nm in one
    # 100 us frame, crowders driven to D/D0 ~ 2.5; demix_msd/msd_full.npz, 2026-09-25).
    if not A.no_chains and A.n_monomers > 0:
        tau_stab = min(tau_stab, 2.0 / (4.0 * D['M'] * S.K_SOFT))
    n_bath = 0 if A.no_bath else int(round(A.bath_eta * V / ((math.pi / 6) * sig['B'] ** 3)))
    n_crowd = int(round(A.phi_crowd * V / ((math.pi / 6) * sig['C'] ** 3)))
    n_mono = 0 if A.no_chains else A.n_monomers
    n_chain = 0
    state = None
    if n_mono and A.state_from:
        state = lattice_state(A.state_from, A.nx)
        n_chain = len(state['chains'])
        if state['n_bound'] > n_mono:
            raise SystemExit('state %s binds %d subunits, more than --n-monomers %d'
                             % (A.state_from, state['n_bound'], n_mono))
        if state['crowders'] is not None and len(state['crowders']) != n_crowd:
            raise SystemExit('state %s holds %d crowders, the scene %d: phi_crowd does not match'
                             % (A.state_from, len(state['crowders']), n_crowd))
    elif n_mono:
        n_chain = A.n_chains if A.n_chains is not None else max(1, n_mono // A.mean_len)
        n_chain = min(n_chain, n_mono // A.mean_len)
    return dict(L=L, V=V, box=[L, L, L], sig=sig, D=D, tau_stab=tau_stab,
                dt=(A.dt if A.dt is not None else tau_stab / 2.5),
                n_bath=n_bath, n_crowd=n_crowd, n_mono=n_mono, n_chain=n_chain, state=state,
                n0=n_bath + n_crowd + n_mono)


def lattice_state(path, nx):
    """The last recorded configuration of a lattice demixing run, in the ReaDDy frame (nm, box centred).

    Chains come from the stored backbones (bb_len / bb_pos, voxel coordinates in backbone order, every
    bond one fcc edge of sqrt(2) h, i.e. at BOND_NM). Crowders come from the last snap_body_<k>, linear
    voxel indices, x fastest; a run without snapshots returns crowders=None and the caller places them
    uniformly. Voxel centres are used, so a body on the h/2 sub-lattice is off by at most h/2.
    """
    d = np.load(path, allow_pickle=True)
    lens = np.asarray(d['bb_len'], int)
    pos = (np.asarray(d['bb_pos'], float) + 0.5) * H - 0.5 * nx * H
    off = np.concatenate([[0], np.cumsum(lens)])
    chains = [pos[off[i]:off[i + 1]] for i in range(len(lens))]
    snaps = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_body_'))
    crowders = None
    if snaps:
        idx = np.asarray(d['snap_body_%d' % snaps[-1]], np.int64)
        xyz = np.stack([idx % nx, (idx // nx) % nx, idx // (nx * nx)], axis=1)
        crowders = (xyz + 0.5) * H - 0.5 * nx * H
    return dict(chains=chains, n_bound=int(lens.sum()), crowders=crowders, path=path,
                t=float(np.asarray(d['times'])[-1]))


def build_system(A, r):
    """The ReaDDy system: species, pair exclusion, chain bonds and (unless static) the chemistry."""
    sig, D = r['sig'], r['D']
    system = readdy.ReactionDiffusionSystem(r['box'], unit_system=None)
    system.kbt = 1.0
    if r['n_bath']:
        system.add_species('B', D['B'])
    system.add_species('C', D['C'])
    if r['n_mono']:
        system.add_species('M', D['M'])
        # Chain beads are TOPOLOGY species: K is an interior bead, Hd a reactive tip.
        system.add_topology_species('K', D['M'])
        system.add_topology_species('Hd', D['M'])
        system.topologies.add_type('Chain')

    names = (['B'] if r['n_bath'] else []) + ['C'] + (['M', 'K', 'Hd'] if r['n_mono'] else [])

    def radius(n):
        return {'B': sig['B'], 'C': sig['C']}.get(n, sig['M'])

    done = set()
    for a in names:
        for b in names:
            key = tuple(sorted((a, b)))
            if key in done:
                continue
            done.add(key)
            if a == b == 'C' and A.no_crowder_pair:
                continue
            system.potentials.add_harmonic_repulsion(
                a, b, S.K_SOFT, S.bh_contact(0.5 * (radius(a) + radius(b))))

    if r['n_mono']:
        for pair in (('K', 'K'), ('K', 'Hd'), ('Hd', 'Hd')):
            system.topologies.configure_harmonic_bond(
                pair[0], pair[1], force_constant=S.K_SOFT, length=BOND_NM)
        if not A.static_chains:
            # GROWTH: a free monomer joins at the tip, which becomes interior. Particle fusion.
            system.topologies.add_spatial_reaction(
                'Grow: Chain(Hd) + (M) -> Chain(K--Hd)', rate=A.k_poly, radius=1.5 * sig['M'])

            # SHRINK: a structural reaction detaching the tip, which becomes free monomer again.
            # This is the half that cannot be written as a spatial reaction, and it is why the
            # chains here are a real topology rather than a frozen string.
            def shrink(topology):
                recipe = readdy.StructuralReactionRecipe(topology)
                verts = topology.get_graph().get_vertices()
                if len(verts) < 3:
                    return recipe
                for v in verts:
                    if len(v.neighbors()) == 1:
                        recipe.separate_vertex(v.particle_index)
                        recipe.change_particle_type(v.particle_index, 'M')
                        break
                return recipe

            system.topologies.add_structural_reaction(
                'Shrink', 'Chain', reaction_function=shrink,
                rate_function=lambda top: A.k_depoly, expect_connected=False)
    return system


def make_simulation(A, system):
    kernel = 'SingleCPU' if A.arm == 'sc' else 'CPU'
    sim = system.simulation(kernel=kernel)
    if A.arm == 'cpu':
        try:
            sim.kernel_configuration.n_threads = int(A.threads)
        except Exception as exc:
            print('WARNING: could not set n_threads: %s' % exc)
    sim.output_file = ''
    return sim


def seed(A, r, sim, rng):
    """Place the bath, the crowders and the chains. Returns the number of beads placed in chains."""
    half = np.array(r['box']) / 2.0
    if r['n_bath']:
        sim.add_particles('B', rng.uniform(-half, half, size=(r['n_bath'], 3)))
    st = r.get('state')
    if st is not None and st['crowders'] is not None:
        sim.add_particles('C', st['crowders'])
    else:
        sim.add_particles('C', rng.uniform(-half, half, size=(r['n_crowd'], 3)))
    n_seeded = 0
    if r['n_mono'] and st is not None:
        # The lattice's own chains. A one-subunit chain is a nucleus, a topology with a single tip.
        for pos in st['chains']:
            n = len(pos)
            types = ['Hd'] if n == 1 else ['Hd'] + ['K'] * (n - 2) + ['Hd']
            top = sim.add_topology('Chain', types, pos)
            for i in range(n - 1):
                top.get_graph().add_edge(i, i + 1)
            n_seeded += n
        left = r['n_mono'] - n_seeded
        if left > 0:
            sim.add_particles('M', rng.uniform(-half, half, size=(left, 3)))
    elif r['n_mono']:
        # Chains laid down as straight strings at the bond length, placed at random and wrapped.
        for _ in range(r['n_chain']):
            origin = rng.uniform(-half, half, size=3)
            step = rng.normal(size=3)
            step /= np.linalg.norm(step)
            step *= BOND_NM
            pos = np.array([origin + i * step for i in range(A.mean_len)])
            for k in range(3):
                pos[:, k] = (pos[:, k] + half[k]) % r['box'][k] - half[k]
            types = ['Hd'] + ['K'] * (A.mean_len - 2) + ['Hd']
            top = sim.add_topology('Chain', types, pos)
            # add_topology places the particles but leaves the graph EDGELESS, and an unconnected
            # topology is rejected at the first step. The backbone has to be wired explicitly.
            for i in range(A.mean_len - 1):
                top.get_graph().add_edge(i, i + 1)
            n_seeded += A.mean_len
        left = r['n_mono'] - n_seeded
        if left > 0:
            sim.add_particles('M', rng.uniform(-half, half, size=(left, 3)))
    return n_seeded


def describe(A, r, n_seeded):
    if r.get('state') is not None:
        return ('[demix-scene] box %.0f nm, N=%d (bath %d, crowder %d, polymer %d, %d bound in %d chains from '
                '%s at t=%.2f s, crowders %s), bond %.2f nm, dt %.3g s (tau_stab %.3g s)'
                % (r['L'], r['n0'], r['n_bath'], r['n_crowd'], r['n_mono'], n_seeded, r['n_chain'],
                   os.path.basename(r['state']['path']), r['state']['t'],
                   'from snapshot' if r['state']['crowders'] is not None else 'uniform',
                   BOND_NM, r['dt'], r['tau_stab']))
    return ('[demix-scene] box %.0f nm, N=%d (bath %d, crowder %d, polymer %d in %d chains of %d), '
            'crowder pair %s, chains %s, bond %.2f nm, dt %.3g s (tau_stab %.3g s)'
            % (r['L'], r['n0'], r['n_bath'], r['n_crowd'], r['n_mono'], r['n_chain'], A.mean_len,
               'OFF' if A.no_crowder_pair else 'ON',
               'static' if A.static_chains else ('none' if A.no_chains else 'reactive'),
               BOND_NM, r['dt'], r['tau_stab']))
