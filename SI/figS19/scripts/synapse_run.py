"""Run one arm of the immune-synapse contact-zone problem on the MPD-RDDME lattice.

    python synapse_run.py --arm excluded --tsim 20 --gpu 0 --out out/exc_theirs.npz

ARMS
    point      hard-core exclusion disarmed. This is the control and it is the argument: it is
               what CRDDME does, and it has no ceiling on the core density.
    excluded   hard-sphere volume exclusion armed, diameters from synapse_model.

WHAT IT WRITES
    An npz with the radial histogram of every species at every frame, the core counts, and the
    metadata needed to reconstruct the run. The .lm is discarded unless --keep-lm is given,
    because a hundred frames of a 288^2 lattice is half a gigabyte and the radial histogram is
    the only reduction any figure needs.

INDEX-ORDER TRAP
    jLM region masks are indexed [x, y, z]. The potential basis the solver wants is flat with x
    running fastest, i.e. a (Nz, Ny, Nx) array ravelled C-order. The transpose between them is
    not optional and is silent if wrong: the field would simply point along the wrong axis and
    the disk would still look like a disk.

SPECIES INDEXING TRAP
    Lattice particle types are 1-based (0 is empty), but setSpeciesCoupling and
    setSpeciesHardSphereDiameters are indexed 0-based. The offset is sp.idx - 1.

AXIS-ORDER TRAP
    jLM REVERSES its axes on the way to the solver. RDME.resizeLattice does
    `nx,ny,nz = dimensions` and then `ByteLattice(nz,ny,nx,...)`, so the LAST entry of the
    dimensions tuple is the axis the MPD kernel calls x, the one that must be divisible by 32.
    Its masks and placeNumber calls follow the same reversed convention. Everything in this
    module is held in solver order (x, y, z) and converted at the two jLM boundaries by
    `to_jlm`. Getting this wrong is not silent for a slab, since the kernel rejects an x
    dimension of 1, but for a cube it would be, which is why the notebook this was built from
    brute-forces the permutation at analysis time instead of knowing it.
"""
import argparse, json, math, os, sys, tempfile, time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synapse_model as M

import lm
from jLM.RDME import Sim as RDMESim
from jLM.RegionBuilder import RegionBuilder
from jLM.units import nm


def to_jlm(a):
    """(x, y, z) in solver order -> the (z, y, x) order jLM's masks and dimensions use."""
    return np.ascontiguousarray(a.transpose(2, 1, 0))


def lattice_dims(h_nm, r_disk_um, nz):
    """Smallest X = Y that covers the disk and satisfies the MPD kernel's X %% 32 == 0."""
    need = int(math.ceil(2.0 * r_disk_um * 1000.0 / h_nm))
    n = int(math.ceil(need / 32.0)) * 32
    return n, n, nz


def radial_grid(nx, ny, nz, h_nm, z_active):
    """Radius in um of every voxel centre from the disk centre, shape (nx, ny, nz).
    Voxels outside the active z-slice are given a radius beyond the disk so every mask that
    thresholds on radius automatically excludes them."""
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    ix = np.arange(nx)[:, None, None]
    iy = np.arange(ny)[None, :, None]
    r = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2) * (h_nm * 1e-3)
    r = np.broadcast_to(r, (nx, ny, nz)).copy()
    if nz > 1:
        off = np.ones(nz, bool); off[z_active] = False
        r[:, :, off] = 1e9
    return r


def pin_mask(r_um, args, h_nm, rng):
    """Immobile membrane-cytoskeleton pinning points, Dharan & Farago type B.

    They are NOT a species here, they are removed lattice sites. That is deliberate. Their
    reference for the spacing is Kusumi's picket-fence review, so the object being represented is
    an actin-anchored transmembrane picket, and Dharan & Farago give each one a whole cell of
    their 100 nm lattice. A real picket is a few nm across, so the excluded area they assign it is
    two orders of magnitude larger than the protein; what their point obstacle really coarse-grains
    is a compartment boundary, not a molecule. A blocked site is the honest lattice analogue of
    that, and the patch size is the parameter that carries the ambiguity.

    They are absent from the actin-depleted central region by construction, as in their model.
    """
    if args.pin_spacing <= 0:
        return np.zeros(r_um.shape, bool)
    m = np.zeros(r_um.shape, bool)
    eligible = (r_um >= args.r_core) & (r_um <= args.r_disk)
    ex, ey, ez = np.where(eligible)
    area_um2 = len(ex) * (h_nm * 1e-3) ** 2
    n_pin = int(round(area_um2 / (args.pin_spacing * 1e-3) ** 2))
    if n_pin <= 0:
        return m
    pick = rng.choice(len(ex), size=min(n_pin, len(ex)), replace=False)
    half = args.pin_size // 2
    nx, ny, nz = r_um.shape
    for i in pick:
        x0, x1 = max(0, ex[i] - half), min(nx, ex[i] - half + args.pin_size)
        y0, y1 = max(0, ey[i] - half), min(ny, ey[i] - half + args.pin_size)
        m[x0:x1, y0:y1, ez[i]] = True
    m &= eligible                                  # never inside the core, never off the disk
    return m


def build(path, args, table, dims, z_active):
    nx, ny, nz = dims
    h_nm = M.H_NM
    D_max = max(t[1] for t in table) * 1e-12                 # um^2/s -> m^2/s
    dt = args.q_stab * (h_nm * 1e-9) ** 2 / D_max
    # Reversed on purpose: jLM's last dimension is the solver's x. See the axis-order trap.
    sim = RDMESim(name='synapse', filename=path, dimensions=[nz, ny, nx],
                  latticeSpacing=nm(h_nm), regionName='out', latticeType='Byte', dt=dt)

    r_um = radial_grid(nx, ny, nz, h_nm, z_active)
    m_core = r_um < args.r_core
    m_mid = (r_um >= args.r_core) & (r_um < args.r_psmac)
    m_rim = (r_um >= args.r_psmac) & (r_um <= args.r_disk)
    rng_pin = np.random.default_rng(args.seed + 7717)
    m_pin = pin_mask(r_um, args, h_nm, rng_pin)
    m_mid = m_mid & ~m_pin
    m_rim = m_rim & ~m_pin
    n_active = int((m_core | m_mid | m_rim).sum())
    n_eligible = int(((r_um >= args.r_core) & (r_um <= args.r_disk)).sum())
    pin_blocked = float(m_pin.sum()) / max(n_eligible, 1)
    out = sim.region('out'); core = sim.region('core')
    mid = sim.region('mid'); rim = sim.region('rim')
    RegionBuilder(sim).compose((out, np.ones((nz, ny, nx), bool)),
                               (core, to_jlm(m_core)), (mid, to_jlm(m_mid)),
                               (rim, to_jlm(m_rim)))
    active = (core, mid, rim)

    # Every transition is closed first, then opened only between active regions. That is what
    # supplies the no-flux boundary of the disk, and it also seals the inert z-slices when
    # nz > 1, so nothing leaks out of the layer.
    sim.transitionRate(None, None, None, sim.diffusionZero)
    species = {}
    for name, D_um2s, couples, _eq, _lat in table:
        sp = sim.species(name)
        dc = sim.diffusionConst('D_' + name, D_um2s * 1e-12)
        for a in active:
            for b in active:
                sim.transitionRate(sp, a, b, dc)
        species[name] = sp

    # Transport-only control: seed the complex directly and define no reaction. The stationary
    # radial profile is then exactly Boltzmann in the ramp, n(r) proportional to exp(-beta Phi),
    # which is an analytic reference for the drift machinery IN THE PRODUCTION GEOMETRY, and it
    # separates the two ways the core can fill in the db gauge: complexes drifting in, versus
    # complexes preferentially FORMING there because K(r) rises toward the centre.
    counts_pre = 0
    if args.preseed_complex > 0.0:
        px, py, pz = np.where(m_core | m_mid | m_rim)
        rng2 = np.random.default_rng(args.seed + 991)
        npre = rng2.poisson(args.preseed_complex * (h_nm * 1e-3) ** 2, size=len(px))
        for i in np.nonzero(npre)[0]:
            sim.placeNumber(species['TCR_pMHC'], int(pz[i]), int(py[i]), int(px[i]), int(npre[i]))
        counts_pre = int(npre.sum())

    # jLM's reaction parser accepts Species instances only, never names. And `regions` is NOT
    # optional despite defaulting to None: with no regions the reaction is assigned to no site
    # type, the location matrix stays empty and the reaction NEVER FIRES, silently. The run
    # completes, conserves mass and produces a perfectly reasonable-looking diffusion-only
    # trajectory with zero product. Asserted below rather than trusted.
    if 'TCR_pMHC' in species and not args.no_reaction:
        sim.reaction([species['TCR'], species['pMHC']], [species['TCR_pMHC']],
                     sim.rateConst('k_on_tcr', M.K_ON_TCR, 2), regions=list(active))
        sim.reaction([species['TCR_pMHC']], [species['TCR'], species['pMHC']],
                     sim.rateConst('k_off_tcr', M.K_OFF_TCR, 1), regions=list(active))
        if len(sim._reactionLocations) != 2 * len(active):
            raise SystemExit('reactions were not assigned to every active region: %r'
                             % (sim._reactionLocations,))
    elif args.no_reaction and sim._reactionLocations:
        raise SystemExit('--no-reaction requested but reactions were assigned')
    # LFA-1-ICAM-1 and CD45 are seeded as preformed, non-reacting occupants. They are present
    # only to exclude volume, which is precisely the role Isaacson & Zhang could not give them.

    # Seeding. Their initial condition is a uniform surface density of free TCR and pMHC
    # OUTSIDE the actin-depletion region, with no complexes anywhere.
    rng = np.random.default_rng(args.seed)
    seed_mask = m_mid | m_rim
    sx, sy, sz = np.where(seed_mask)
    n_vox = len(sx)
    voxel_area_um2 = (h_nm * 1e-3) ** 2
    per_vox = args.density * voxel_area_um2
    counts = {}
    for name in ('TCR', 'pMHC'):
        if name not in species or args.preseed_complex > 0.0:
            continue
        n = rng.poisson(per_vox, size=n_vox)
        for i in np.nonzero(n)[0]:
            sim.placeNumber(species[name], int(sz[i]), int(sy[i]), int(sx[i]), int(n[i]))
        counts[name] = int(n.sum())
    # The second adhesion pair, when present, is seeded uniformly over the whole disk: it is
    # the "other synapse component" whose repulsion Isaacson & Zhang had to delete.
    if 'LFA_ICAM' in species:
        ax, ay, az = np.where(m_core | m_mid | m_rim)
        n = rng.poisson(args.density_lfa * voxel_area_um2, size=len(ax))
        for i in np.nonzero(n)[0]:
            sim.placeNumber(species['LFA_ICAM'], int(az[i]), int(ay[i]), int(ax[i]), int(n[i]))
        counts['LFA_ICAM'] = int(n.sum())
    if 'CD45' in species:
        ax, ay, az = np.where(m_core | m_mid | m_rim)
        n = rng.poisson(args.density_cd45 * voxel_area_um2, size=len(ax))
        for i in np.nonzero(n)[0]:
            sim.placeNumber(species['CD45'], int(az[i]), int(ay[i]), int(ax[i]), int(n[i]))
        counts['CD45'] = int(n.sum())

    if counts_pre:
        counts['TCR_pMHC'] = counts_pre
    counts['_pin_voxels'] = int(m_pin.sum())
    counts['_pin_blocked_fraction'] = round(pin_blocked, 5)
    steps = int(round(args.tsim / dt))
    per_frame = max(1, steps // args.frames)
    sim.timestep = dt
    sim.simulationTime = args.tsim
    # jLM writes these straight into the file as integers: they are STEP COUNTS, not times.
    sim.latticeWriteInterval = int(per_frame)
    sim.speciesWriteInterval = int(per_frame)
    return sim, species, dt, steps, per_frame, r_um, counts


def potential_basis(r_um, dims, r_psmac, r_disk):
    """psi0 = the Isaacson-Zhang ramp shape in um, flattened in the kernel's (z, y, x) order."""
    nx, ny, nz = dims
    r = np.clip(r_um, 0.0, r_disk)
    shape = np.where(r <= r_psmac, r, r_psmac + 2.0 * (r - r_psmac))
    psi = np.ascontiguousarray(shape.transpose(2, 1, 0).ravel(order='C').astype(np.float32))
    return psi, np.zeros(nx * ny * nz, np.float32)


def configure(solver, args, table, species, psi0, psi1, h_nm):
    solver.setPotentialBasis(psi0, psi1)
    n_sp = max(sp.idx for sp in species.values())
    diam = np.zeros(n_sp, np.float64)
    for name, _D, _couples, _eq, sig_lat in table:
        i = species[name].idx - 1                       # lattice types are 1-based
        w = M.coupling_weight(name, args.gauge)
        solver.setSpeciesCoupling(i, float(args.f0) * w, 0.0)
        diam[i] = sig_lat
    if args.arm == 'excluded':
        solver.setSpeciesHardSphereDiameters(diam, float(h_nm ** 3))
        solver.setHardCoreExclusion(1.0, 2)
    solver.setHopIntegration(1)
    # Adaptive tau rescales propensities. The clock and the rates have come apart here before
    # (rates 18.3x too low), so the scale is reported at the end of every run and the reaction
    # rate is gated against the analytic initial rate rather than assumed.
    if args.adaptive_tau:
        solver.setAdaptiveTau(1, 0.2, 0.05)
    solver.setLatticeWordBound(1)
    solver.setLatticeTransportBound(1)
    solver.validateStability()
    return diam


def reduce_frames(path, species, r_um, args, dims):
    """Radial histogram per species per frame, plus the count inside the core, plus (when
    --map-times is given) the 2D occupancy map of the active layer at the requested times.

    The map is what reproduces the Isaacson-Zhang figure-8 panel. It is stored as counts per
    voxel; the figure script converts to surface density, because their mesh cell is 7093 nm^2
    and our voxel is 1600 nm^2, so raw counts are not comparable between the two papers."""
    edges = np.arange(0.0, args.r_disk + 1e-9, args.dr)
    r_flat = r_um.ravel()
    inside = r_flat <= args.r_disk
    idx = np.digitize(r_flat, edges) - 1
    nb = len(edges) - 1
    ok = inside & (idx >= 0) & (idx < nb)
    # Voxels per bin: the normalization that turns a count into a surface density.
    vox_per_bin = np.bincount(idx[ok], minlength=nb).astype(float)
    names = list(species.keys())
    maps, map_times, map_frames = [], [], set()
    with h5py.File(path, 'r') as f:
        grp = f['/Simulations/0000001']
        lat = grp['Lattice']
        keys = sorted(lat.keys())
        times = grp['LatticeTimes'][...][:len(keys)]
        if args.map_times:
            want = [float(x) for x in args.map_times.split(',')]
            map_frames = {int(np.argmin(np.abs(times - w))) for w in want}
        hist = np.zeros((len(keys), len(names), nb))
        core = np.zeros((len(keys), len(names)))
        total = np.zeros((len(keys), len(names)))
        maxocc = np.zeros(len(keys))
        for fi, k in enumerate(keys):
            block = lat[k][...]                          # (nx, ny, nz, pps)
            occ_all = np.zeros(block.shape[:3], np.int32)
            for si, name in enumerate(names):
                occ = (block == species[name].idx).sum(-1).astype(np.int32)
                occ_all += occ
                o = occ.ravel()
                hist[fi, si] = np.bincount(idx[ok], weights=o[ok], minlength=nb)
                core[fi, si] = o[inside & (r_flat < args.r_core)].sum()
                total[fi, si] = o[inside].sum()
            maxocc[fi] = occ_all.max()
            if fi in map_frames:
                sp_map = species[args.map_species].idx
                occ2 = (block == sp_map).sum(-1)
                # The active layer is one voxel thick; sum over the z axis to collapse it. The
                # axis of length nz is located by shape, not assumed: nx == ny here, so an x/y
                # transpose is invisible in a radially symmetric problem but a z/x confusion is
                # not, and it would silently discard most of the lattice.
                zax = [a for a, n in enumerate(occ2.shape) if n == dims[2]]
                if len(zax) != 1:
                    raise SystemExit('cannot identify the z axis in lattice shape %r against '
                                     'dims %r' % (occ2.shape, dims))
                maps.append(occ2.sum(axis=zax[0]).astype(np.float32))
                map_times.append(float(times[fi]))
    out = dict(edges=edges, times=times, hist=hist, core=core, total=total,
               vox_per_bin=vox_per_bin, names=np.array(names), maxocc=maxocc)
    if maps:
        out['maps'] = np.stack(maps)
        out['map_times'] = np.array(map_times)
        out['map_species'] = np.array(args.map_species)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--arm', choices=('point', 'excluded'), default='excluded')
    p.add_argument('--species', choices=('minimal', 'full'), default='minimal')
    p.add_argument('--density', type=float, default=M.SIGMA_SURF_THEIRS,
                   help='free TCR and pMHC surface density, per um^2, seeded outside the core')
    p.add_argument('--density-lfa', type=float, default=23.0)
    p.add_argument('--density-cd45', type=float, default=21.0)
    p.add_argument('--f0', type=float, default=M.F0_KT_PER_UM)
    p.add_argument('--gauge', choices=M.GAUGES, default='db',
                   help='see the GAUGES note in synapse_model')
    p.add_argument('--tsim', type=float, default=20.0, help='physical seconds')
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--q-stab', type=float, default=0.008,
                   help='dt = q h^2 / D_max; 0.008 is the value the working motor notebook uses')
    p.add_argument('--r-disk', type=float, default=M.R_DISK_UM)
    p.add_argument('--r-core', type=float, default=M.R_CORE_UM)
    p.add_argument('--r-psmac', type=float, default=M.R_PSMAC_UM)
    p.add_argument('--dr', type=float, default=0.1, help='radial bin width, um')
    p.add_argument('--nz', type=int, default=8,
                   help='z extent. The MPD kernel rejects a z dimension not divisible by 8, so a '
                        'genuine single-voxel layer is impossible; 8 slices are allocated and all '
                        'but the middle one are sealed into the inert `out` region, which makes '
                        'the active layer exactly one voxel thick at the cost of empty slices.')
    p.add_argument('--packing-match', action='store_true',
                   help='use the jamming-limit diameter convention instead of virial matching')
    p.add_argument('--adaptive-tau', type=int, default=0,
                   help='OFF by default and it must stay off. With it on the solver drives the '
                        'tau scale to its floor and the chemistry then advances about 1/scale '
                        'times too fast per unit REPORTED clock: at scale 0.05 the well-mixed '
                        'gate reads 18x the correct product at early times. With it off the same '
                        'gate reads 0.99. The bare timestep at q_stab 0.008 is already safe here '
                        '(hop probability 0.008, reaction probability 3e-4 per step).')
    p.add_argument('--no-reaction', action='store_true',
                   help='define no reaction; use with --preseed-complex')
    p.add_argument('--preseed-complex', type=float, default=0.0,
                   help='seed the complex uniformly at this surface density and seed no '
                        'free species: the transport-only control')
    p.add_argument('--pin-spacing', type=float, default=0.0,
                   help='mean spacing in nm of immobile pinning obstacles outside the '
                        'core; 0 disables them. Dharan and Farago use 300 nm.')
    p.add_argument('--pin-size', type=int, default=3,
                   help='side of the blocked voxel patch. 3 at 40 nm is 120 nm, the '
                        'closest to the 100 nm cell Dharan and Farago block; 1 is the '
                        'smallest obstacle this lattice can represent at all.')
    p.add_argument('--map-times', default='',
                   help='comma-separated times in s at which to store the 2D map of\n                        the active layer; the nearest written frame is used')
    p.add_argument('--map-species', default='TCR_pMHC')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--keep-lm', default='')
    p.add_argument('--out', required=True)
    A = p.parse_args()

    names = M.MINIMAL if A.species == 'minimal' else None
    table = M.species_table(names, M.H_NM, virial_match=not A.packing_match)
    dims = lattice_dims(M.H_NM, A.r_disk, A.nz)
    z_active = A.nz // 2
    print('lattice %dx%dx%d @ %.0f nm  (%.2f um across)  active z-slice %d'
          % (dims + (M.H_NM, dims[0] * M.H_NM * 1e-3, z_active)))
    print('gauge=%s  arm=%s  f0=%.3g kT/um' % (A.gauge, A.arm, A.f0))
    for name, D, _couples, eq, lat in table:
        print('  %-10s D=%.3f um^2/s  gamma=%.3g kT/um  equiv %.2f nm  lattice %.2f nm'
              % (name, D, A.f0 * M.coupling_weight(name, A.gauge), eq, lat))
    sys.stdout.flush()

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        path = A.keep_lm or os.path.join(td, 'synapse.lm')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        sim, species, dt, steps, per_frame, r_um, counts = build(path, A, table, dims, z_active)
        print('dt=%.3e s  steps=%d  frames every %d steps  seeded %s'
              % (dt, steps, per_frame, counts)); sys.stdout.flush()
        solver = lm.DriftMpdRdmeSolver()
        psi0, psi1 = potential_basis(r_um, dims, A.r_psmac, A.r_disk)
        diam = configure(solver, A, table, species, psi0, psi1, M.H_NM)
        sim.finalize()
        t_run = time.time()
        sim.run(solver=solver, cudaDevices=[A.gpu])
        wall = time.time() - t_run
        try:
            tau_scale = float(solver.getTauScale())
        except Exception:
            tau_scale = float('nan')
        print('tau_scale = %.6g' % tau_scale)
        viol = int(solver.getLatticeWordBoundViolations())
        print('run %.1f s wall; word mark %d, violations %d'
              % (wall, int(solver.getLatticeMaxWord()), viol)); sys.stdout.flush()
        if viol:
            raise SystemExit('word-bound invariant violated %d times: results unsafe' % viol)
        red = reduce_frames(path, species, r_um, A, dims)

    meta = dict(arm=A.arm, gauge=A.gauge, species_set=A.species, density=A.density, f0=A.f0, tsim=A.tsim,
                dt=dt, steps=steps, h_nm=M.H_NM, dims=list(dims), nz=A.nz, seed=A.seed,
                r_disk=A.r_disk, r_core=A.r_core, r_psmac=A.r_psmac,
                diameters=diam.tolist(), seeded=counts, wall_s=wall,
                packing_match=bool(A.packing_match), q_stab=A.q_stab,
                tau_scale=tau_scale, adaptive_tau=int(A.adaptive_tau),
                no_reaction=bool(A.no_reaction), preseed_complex=A.preseed_complex,
                pin_spacing=A.pin_spacing, pin_size=A.pin_size)
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    np.savez_compressed(A.out, meta=json.dumps(meta), **red)
    print('wrote %s   (total %.1f s)' % (A.out, time.time() - t0))
    print('max total occupancy over frames: %d of %d slots'
          % (int(red['maxocc'].max()), lm.getCompiledLatticeMaxOccupancy()))


if __name__ == '__main__':
    main()
