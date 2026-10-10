#!/usr/bin/env python3
"""Fixed-length semiflexible chain on the structure handler.

Everything the conformational and dynamic measurements need, with the chemistry off so the chain
length cannot change. No formation channel is armed and no tip driver is called, so the only things
that act are the interior conformational moves and, when explicitly switched on, rigid translation
and rigid rotation of the whole complex.

TWO BOND ARMS. The free arm leaves the stretch term off, so bonds are drawn from all twenty six
neighbours with lengths one, root two and root three. The edge arm sets the rest length to root two
with a large stretch stiffness, which selects the twelve edge diagonals. That matters for more than
tidiness: with one bond length the stretch energy is the same on every bond, so it is an additive
constant and cancels from every conformational energy difference, and the bending stiffness becomes
the only parameter. With mixed lengths the two terms are not independent.

The stiffness needed for clean selection is set by the NARROWEST length gap the selection has to
open. For the edge orbit that is the gap to the body diagonals, which is narrower than the gap to
the face directions, so edge selection converges later than face selection does. Four hundred is the
working value; two hundred leaves parts in a hundred thousand of body-orbit weight.

SEEDING. Three seeds. 'exact' draws from the closed-form FCC distribution (fcc_exact.seed): the
turn angles are i.i.d. on this bond set, so a chain grown from the one-joint Boltzmann law is an
exact phantom sample, and one that happens not to intersect itself is an exact SELF-AVOIDING
sample. Use it for anything that reports an equilibrium value, because it removes the equilibration
problem rather than bounding it. 'coil' is a UNIFORM self-avoiding walk, not a Boltzmann sample, so
it is a starting point and never a measurement. 'straight' is the slow side of the equilibration
bracket: only end bond rotation can create curvature and it acts at the tips, so a straight seed
relaxes by transporting curvature inward at a cost that grows with chain length.

UNITS. The single-subunit diffusion coefficient is given here in square micrometres per second and
converted to square nanometres per second before it reaches the engine, which is the unit the hop
and rotation setters expect alongside a voxel edge in nanometres.

TIME. The interior kernel is launched once per lattice timestep in a full simulation, so one step of
the standalone driver is one lattice timestep. Physical time is the step count times the timestep
times the time scale factor. Using the bare timestep is wrong whenever adaptive stepping is armed.
"""
import numpy as np

import chain_geometry as G
import fcc_exact as FX

DIR_NONE = 255
ROLE_BODY = 0
FLAG_ACTIVE = 1

ARMS = {
    'free': {'k_stretch': 0.0, 'rest_len': 1.0, 'orbit': None},
    'edge': {'k_stretch': 400.0, 'rest_len': float(np.sqrt(2.0)), 'orbit': 'edge'},
    'face': {'k_stretch': 400.0, 'rest_len': 1.0, 'orbit': 'face'},
}

MOVE_END_BOND, MOVE_KINK, MOVE_CRANK = 1, 2, 4
MOVE_ALL = MOVE_END_BOND | MOVE_KINK | MOVE_CRANK


def physical_time(n_steps, tau_s, tau_scale):
    """Steps to seconds. The time scale factor is not optional: a propensity-scaled step that
    advances the clock by the bare timestep is the accounting error that made every rate wrong by a
    large factor once already."""
    return float(n_steps) * float(tau_s) * float(tau_scale)


def orbit_dirs(orbit):
    """L26 indices belonging to one bond orbit, in the engine's own direction numbering."""
    lens = np.linalg.norm(G.VECS, axis=1)
    if orbit == 'face':
        mask = np.isclose(lens, 1.0)
    elif orbit == 'edge':
        mask = np.isclose(lens, np.sqrt(2.0))
    elif orbit == 'body':
        mask = np.isclose(lens, np.sqrt(3.0))
    else:
        mask = np.ones(26, bool)
    return np.flatnonzero(mask)


def self_avoiding_walk(n, lx, ly, lz, rng, orbit=None, margin=3, tries=4000):
    """A self-avoiding walk of n subunits, optionally restricted to one bond orbit.

    Used as the equilibrated-side seed. It is a uniform walk, not a Boltzmann sample, so it is a
    starting point for equilibration and never itself a measurement.
    """
    allowed = orbit_dirs(orbit)
    for _ in range(tries):
        pos = [np.array([lx // 2, ly // 2, lz // 2], np.int64)]
        occ = {tuple(pos[0])}
        dirs = []
        ok = True
        for _ in range(n - 1):
            for _try in range(200):
                d = int(rng.choice(allowed))
                p = pos[-1] + G.VECS[d]
                if (tuple(p) not in occ
                        and margin <= p[0] < lx - margin
                        and margin <= p[1] < ly - margin
                        and margin <= p[2] < lz - margin):
                    break
            else:
                ok = False
                break
            pos.append(p)
            occ.add(tuple(p))
            dirs.append(d)
        if ok:
            return np.array(pos, np.int64), np.array(dirs, np.int64)
    raise RuntimeError("could not build a self-avoiding walk of %d subunits" % n)


def straight_walk(n, lx, ly, lz, orbit=None):
    """A straight chain, used for the slow side of the equilibration bracket."""
    d = int(orbit_dirs(orbit)[0]) if orbit else G.l26_index(1, 0, 0)
    step = G.VECS[d]
    start = np.array([lx // 2, ly // 2, lz // 2], np.int64) - step * (n // 2)
    pos = np.array([start + step * i for i in range(n)], np.int64)
    return pos, np.full(n - 1, d, np.int64)


def _flags(species):
    """Registry flags with the species tag in bits 8..15, which is not optional.

    The move kernels read `speciesType = (flags >> 8) & 0xFF` and every species-keyed term is
    skipped when it is zero: the external potential (structCoupledPhi), the membrane adhesion field
    (structMembAdhDelta) and the White-Bear work against the bath (structVexInsertDelta). A chain
    registered without a species is therefore INVISIBLE TO THE BATH, silently — eta 0 and eta 0.15
    return bit-identical trajectories with hundreds of thousands of crowders in the box. That is
    what this seeder used to do. There is deliberately no way to register species 0 any more.
    """
    sp = int(species)
    if not 1 <= sp <= 255:
        raise ValueError("species tag must be 1..255 (it goes in bits 8..15 of the registry flags); "
                         "0 would make the chain invisible to the bath and to every field")
    return FLAG_ACTIVE | (sp << 8)


def seed_chain(sol, pos, dirs, lx, ly, lz, species, fill=1):
    """Write one chain into the overlay and register it.

    Registry count is subunits times fill, which is what the hop and rotation drivers check. A
    registry that disagrees with the occupancy makes those drivers abort, so it is set consistently
    here even when only the interior driver will run.
    """
    xyz = lx * ly * lz
    l1 = np.zeros(xyz, np.int32)
    l2 = np.full(xyz, DIR_NONE, np.int32)
    l3 = np.zeros(xyz, np.int32)
    vox = [int(p[0]) + lx * int(p[1]) + lx * ly * int(p[2]) for p in pos]
    if len(set(vox)) != len(vox):
        raise ValueError("seed positions are not distinct")
    for i, v in enumerate(vox):
        l1[v] = 1
        l3[v] = (ROLE_BODY << 4) | fill
        if i < len(vox) - 1:
            l2[v] = int(dirs[i])
    sol.setStructureOverlay(l1, l2, l3)
    sol.setStructureRegistry(1, np.array([len(vox) * fill], np.int32),
                             pos.astype(np.float64).mean(axis=0),
                             np.array([_flags(species)], np.int32))
    sol.setStructureTips(0, vox[0], vox[-1], -1)
    return vox


def seed_chains(sol, chains, lx, ly, lz, species, fill=1):
    """Write K chains into the overlay and register them, one complex id each.

    WHY MORE THAN ONE. The simulation RNG seed does not reach the structure kernels at all:
    struct_step_seed (structure_host_internal.cuh) is keyed on the timestep counter and a channel
    salt, so two runs with the same starting conformation are bit-identical in every bead. The move
    kernels do mix the COMPLEX ID into that hash, so K bodies in one box are K independent streams
    in one run. That is the only way to get an ensemble out of this path without changing the engine
    and breaking every byte-identity baseline. See the note in fcc_exact.seed_many.
    """
    xyz = lx * ly * lz
    l1 = np.zeros(xyz, np.int32)
    l2 = np.full(xyz, DIR_NONE, np.int32)
    l3 = np.zeros(xyz, np.int32)
    counts, com_flat, flags, all_vox = [], [], [], []
    for cid, (pos, dirs) in enumerate(chains, start=1):
        vox = [int(p[0]) + lx * int(p[1]) + lx * ly * int(p[2]) for p in pos]
        if len(set(vox)) != len(vox):
            raise ValueError("chain %d seed positions are not distinct" % cid)
        for i, v in enumerate(vox):
            if l1[v]:
                raise ValueError("chain %d overlaps chain %d at voxel %d" % (cid, l1[v], v))
            l1[v] = cid
            l3[v] = (ROLE_BODY << 4) | fill
            if i < len(vox) - 1:
                l2[v] = int(dirs[i])
        counts.append(len(vox) * fill)
        com_flat.extend(pos.astype(np.float64).mean(axis=0).tolist())
        flags.append(_flags(species))
        all_vox.append(vox)
    sol.setStructureOverlay(l1, l2, l3)
    sol.setStructureRegistry(len(chains), np.array(counts, np.int32),
                             np.array(com_flat, np.float64), np.array(flags, np.int32))
    for k, vox in enumerate(all_vox):
        sol.setStructureTips(k, vox[0], vox[-1], -1)
    return all_vox


def build(n, kappa, arm='edge', seed_kind='coil', rng=None, lx=32, ly=32, lz=32,
          sigma_nm=10.0, voxel_nm3=1000.0, species=1, attempts=None, move_mask=MOVE_ALL,
          rigid_translation=False, rigid_rotation=False,
          d1_um2=0.3, tau_s=50e-6, lambda_nm=10.0):
    """A solver holding one fixed-length chain, with the chemistry off.

    The attempt budget defaults to the chain's bond count rather than a constant, because the kernel
    budgets attempts per COMPLEX and not per monomer, so a constant budget gives every monomer an
    attempt rate that falls as the chain lengthens. Pass an explicit value to reproduce the
    as-shipped constant-budget behaviour.
    """
    import lm
    rng = rng or np.random.default_rng(0)
    cfg = ARMS[arm]
    sol = lm.DriftMpdRdmeSolver()
    sol.setStructureHandler(1, sigma_nm, voxel_nm3)
    sol.configureStructureLattice(lx, ly, lz)

    if seed_kind == 'straight':
        pos, dirs = straight_walk(n, lx, ly, lz, cfg['orbit'])
    elif seed_kind == 'coil':
        pos, dirs = self_avoiding_walk(n, lx, ly, lz, rng, cfg['orbit'])
    elif seed_kind == 'exact':
        if cfg['orbit'] != 'edge':
            raise ValueError("the exact seed is derived for the FCC (edge) bond set only; "
                             "arm '%s' has no closed-form sampler" % arm)
        pos, dirs = FX.seed(n, kappa, lx, ly, lz, rng)
    else:
        raise ValueError("seed_kind must be 'straight', 'coil' or 'exact'")
    seed_chain(sol, pos, dirs, lx, ly, lz, species=species)

    att = int(attempts) if attempts is not None else max(1, n - 1)
    sol.setStructureLaunchParams(att, float(kappa))
    sol.setStructureStretch(cfg['k_stretch'], cfg['rest_len'])
    sol.setStructureInteriorMoveMask(int(move_mask))
    # UNITS. Both setters take the single-subunit diffusion coefficient in SQUARE NANOMETRES per
    # second, with the voxel edge in nanometres, because the kernel forms the hop probability as
    # D(N) times the timestep over the squared edge and needs one length unit throughout. Passing
    # square micrometres is not an error at any level the binding can see: the hop probability comes
    # out a million times too small and the complex simply never moves, with a centre of mass
    # trajectory that is written every step and never changes.
    d1_nm2 = float(d1_um2) * 1e6
    if rigid_translation:
        sol.setStructureSizeDiffusion(0, d1_nm2, tau_s, lambda_nm)   # Rouse D(N) = D1/N
    if rigid_rotation:
        sol.setStructureRotationModel(1, d1_nm2, tau_s, lambda_nm)   # Tirado rod

    return {'sol': sol, 'n': n, 'kappa': float(kappa), 'arm': arm, 'attempts': att,
            'lx': lx, 'ly': ly, 'lz': lz, 'tau_s': tau_s,
            'k_stretch': cfg['k_stretch'], 'rest_len': cfg['rest_len'], 'orbit': cfg['orbit'],
            'rigid_translation': rigid_translation, 'rigid_rotation': rigid_rotation,
            'seed_kind': seed_kind, 'species': int(species)}


def advance(h, n_steps, seed0):
    """Advance the chain. Interior moves always; rigid motion only when the arm switched it on.

    The rigid drivers write a per-step trajectory, the centre of mass for translation and the
    principal axis for rotation, which is returned so the displacement measurements do not have to
    reconstruct it from the overlay. Interior moves have no such output: their effect on the centre
    of mass has to be read from the backbone, which is the whole point of separating the two.
    """
    sol = h['sol']
    out = {}
    sol.runStructureInteriorSteps(n_steps, seed0, h['kappa'], h['attempts'])
    if h['rigid_translation']:
        traj = np.zeros(3 * n_steps, np.float64)
        sol.runStructureHopSteps(n_steps, (seed0 ^ 0x5A5A5A) & 0xFFFFFFFF, traj)
        out['com'] = traj.reshape(n_steps, 3)
    if h['rigid_rotation']:
        axis = np.zeros(3 * n_steps, np.float64)
        sol.runStructureRotationSteps(n_steps, (seed0 ^ 0xA5A5A5) & 0xFFFFFFFF, axis)
        out['axis'] = axis.reshape(n_steps, 3)
    return out


def read(h):
    """Ordered backbone of the single chain, with a hard failure on a corrupt walk."""
    ch = G.read_chains(h['sol'], h['lx'], h['ly'], h['lz']).get(1)
    if ch is None or not ch['ok']:
        raise RuntimeError("backbone readout failed: %s" % (ch or {}).get('reason', 'walk broke'))
    return ch


def sample(h, n_frames, steps_per_frame, seed0, burn_frames=0, p_max=0, max_sep=0):
    """Run and collect per-frame observables, asserting the chain length never changes.

    The length assertion is the guard that makes this a fixed-length measurement rather than a claim
    about one. A chain that silently loses a subunit reports a smaller end to end distance that
    looks entirely reasonable.
    """
    n0 = read(h)['n']
    frames = []
    for f in range(burn_frames + n_frames):
        advance(h, steps_per_frame, (seed0 + 7919 * f) & 0xFFFFFFFF)
        ch = read(h)
        if ch['n'] != n0:
            raise RuntimeError("chain length changed from %d to %d at frame %d"
                               % (n0, ch['n'], f))
        if f < burn_frames:
            continue
        rec = G.observables(ch)
        rec['step'] = (f + 1 - burn_frames) * steps_per_frame
        rec['time_s'] = physical_time(rec['step'], h['tau_s'], h['sol'].getTauScale())
        rec['dirs'] = ch['dirs'].copy()
        if p_max:
            rec['modes'] = G.rouse_modes(ch, p_max)
        if max_sep:
            rec['bond_corr'] = G.bond_correlation(ch, max_sep)
        frames.append(rec)
    return frames


def bond_length_histogram(frames):
    """Realised bond length distribution, used to confirm an orbit arm is actually selected."""
    all_len = np.concatenate([f['bond_lengths'] for f in frames])
    out = {}
    for name, target in (('face', 1.0), ('edge', np.sqrt(2.0)), ('body', np.sqrt(3.0))):
        out[name] = float(np.isclose(all_len, target).mean())
    out['mean'] = float(all_len.mean())
    out['mean_sq_over_sq_mean'] = float((all_len ** 2).mean() / all_len.mean() ** 2)
    return out
