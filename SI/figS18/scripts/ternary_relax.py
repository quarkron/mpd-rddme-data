"""MPD-RDDME relaxation of a TERNARY mixture with an oversize species: the lattice arm of the ternary figure.

A (2 nm) and B (4 nm) are byte-lattice species at eta_A = eta_B = 0.07; C (20 nm) is a compact depositing body on the
structure lattice, one per (80 nm)^3 = 64 in the 32^3 box at h = 10 nm, the same number density as the ReaDDy reference
(readdy_ref/readdy_three.py). Bath statistics are recorded per voxel per frame and every C's centre of mass with them,
so the analysis can bin the bath around C in C's own frame.

Geometry and clock are the binary figure's (vex_relax2.py): a stripe of period eight voxels along x, filled x % 8 < 4,
one period across the ReaDDy box and four across the lattice, so the slowest decaying mode is the same and the two
relax on a common clock tau = D_A t / h^2, D_A = 1e5 nm^2/s. C starts at the centre of the empty stripes,
x in {6, 14, 22, 30}, y and z in {4, 12, 20, 28}: the ReaDDy voxel (6, 4, 4) tiled four times per axis.

The filled voxels are seeded from the reference's tau = 0 joint (n_A, n_B) histogram (--seed-hist, make_seed_hist.py),
so P(n_A, n_B, 0) agrees by construction; --uniform scatters multinomially instead, for smoke tests only.

Three arms:
  model    hard core + smear + reciprocal + collisional friction (auto) + the FREE-VOLUME functional (the fix)
  oldfunc  the same with the free-volume functional off: the body's weighted densities added to the bath's
  point    no exclusion anywhere: ideal bath, C invisible to the bath and free-diffusing (the point-particle limit)

Species and types: 'A' = 1 (primary bath), 'FILL' = 2 (the body's own byte-lattice fill, no particles), 'B2' = 3 (B),
through bath_species.py. C is structure species tag 2 (RIBO_ID), unrelated to byte-lattice types.

The lattice caps a voxel at 64 particles; the slab start doubles the local density, which is why eta is 0.07 and not
the binary figure's 0.10 (at 0.10 the filled-voxel tail reaches 79; readdy_three.py docstring).

Run in the lm2.6 container with PYTHONPATH on the built module, from testing/engine:
  python ternary_relax.py --arm model --seed 0 --seed-hist /path/rt3_seed_joint.npy --out /path/lat3_model_s0.npz
"""
import argparse, json, os, sys, tempfile, time, math
import numpy as np
import h5py
import lm
from pyLM import RDME
from pyLM.units import nm, micron, microsecond
import ribosome_blob as rb
import bath_species as bsp
import bath_scene as scene

NM = 10.0; V = NM ** 3; TAU_S = 50e-6
D_A_UM2 = 0.1                       # 1e5 nm^2/s = 1e-13 m^2/s, the clock (readdy_three.py --d-a)
D1_UM2 = 0.9                        # base D the Kirkwood size model assigns a one-subunit body; C is scaled from it
SIGMA_FILL = 5.0; LAM_HC = 1.0; PERIOD = 8; DUTY = 0.5
BATH_ID = 1; RIBO_ID = 2; D_MODEL_KIRKWOOD = 3
DIR_NONE, ROLE_BODY, FLAG_ACTIVE, FLAG_INERT = 255, 0, 1, 0x10

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--arm', choices=('model', 'oldfunc', 'point'), default='model')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n', type=int, default=32)
ap.add_argument('--dims', default=None,
                help='NX,NY,NZ in voxels; default is the cubic --n. Non-cubic is for the cross-engine '
                     'timing arm, where the box must match what the particle engine can run.')
ap.add_argument('--sigma-a', type=float, default=2.0)
ap.add_argument('--sigma-b', type=float, default=4.0)
ap.add_argument('--sigma-c', type=float, default=20.0)
ap.add_argument('--eta-a', type=float, default=0.07)
ap.add_argument('--eta-b', type=float, default=0.07)
ap.add_argument('--tau-max', type=float, default=4.0)
ap.add_argument('--frames', type=int, default=40, help='target frames over tau_max (adaptive tau adds more)')
ap.add_argument('--seed-hist', default=None, help='npy joint (n_A, n_B) histogram of the reference at tau 0 (filled half)')
ap.add_argument('--uniform', action='store_true', help='multinomial slab start instead of --seed-hist (smoke tests)')
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--tau-us', dest='tau_us', type=float, default=50.0, help='nominal timestep (us); 50 = the figure')
ap.add_argument('--fixed-tau', dest='fixed_tau', action='store_true', help='adaptive tau OFF: every step is --tau-us (tau ladders)')
ap.add_argument('--order', choices=('fixed', 'perm'), default='fixed',
                help='perm: the step operators in a timestep-seeded random order (setStepOrderRandomize, rand-split build)')
ap.add_argument('--smear-after', dest='smear_after', action='store_true',
                help="the bodies' smear field re-stamped after the structure moves (setStructureSmearAfterMoves)")
ap.add_argument('--out', required=True)
A = ap.parse_args()
TAU_S = A.tau_us * 1e-6
if not A.uniform and not A.seed_hist:
    sys.exit('give --seed-hist (make_seed_hist.py) or --uniform')

NX, NY, NZ = (tuple(int(v) for v in A.dims.split(',')) if A.dims else (A.n, A.n, A.n))
N = NX                       # kept: the cubic runs and every message below read N
XYZ = NX * NY * NZ
sig = {'A': A.sigma_a, 'B': A.sigma_b, 'C': A.sigma_c}
D_UM2 = {'A': D_A_UM2, 'B': D_A_UM2 * sig['A'] / sig['B'], 'C': D_A_UM2 * sig['A'] / sig['C']}     # Stokes scaling
NA = int(round(A.eta_a * XYZ * V / (np.pi / 6 * sig['A'] ** 3)))
Bspec = bsp.parse([f"{sig['B']}:{A.eta_b}:{D_UM2['B']}"], D_UM2['C'], sig['C'])
t_max = A.tau_max * NM ** 2 / (D_A_UM2 * 1e6)                     # s
n_steps = int(math.ceil(t_max / TAU_S)); per_frame = max(1, n_steps // A.frames)
rng = np.random.default_rng(A.seed)
excl = A.arm != 'point'

# C bodies: centres of the empty stripes, the ReaDDy voxel (6, 4, 4) tiled
cx = [PERIOD * k + int(round(PERIOD * (1 + DUTY) / 2)) for k in range(NX // PERIOD)]
cy = [PERIOD * k + PERIOD // 2 for k in range(NY // PERIOD)]
cz = [PERIOD * k + PERIOD // 2 for k in range(NZ // PERIOD)]
centers = [(x, y, z) for x in cx for y in cy for z in cz]
vox = lambda x, y, z: z * (NX * NY) + y * NX + x                   # the solver's linear index: x fastest


def stripe_lattice(path, joint):
    """Write the striped initial lattice into the saved .lm: filled x % PERIOD < DUTY*PERIOD, pairs (n_A, n_B) from
    the reference's joint histogram (or multinomial). A = type 1, B = type 3. Refuses to exceed the slot cap."""
    with h5py.File(path, 'r+') as f:
        lat = f['/Model/Diffusion/Lattice']; shape = lat.shape; max_occ = shape[3]
        arr = np.zeros(shape, np.uint8)
        xs = np.arange(shape[0]); live = xs[(xs % PERIOD) < int(round(DUTY * PERIOD))]
        for (x, y, z) in centers:
            assert x not in live, 'a body sits in a filled stripe'
        sites = np.array([(x, y, z) for x in live for y in range(shape[1]) for z in range(shape[2])]); ns = len(sites)
        if joint is not None:
            J = np.asarray(joint, float); J = J / J.sum()
            flat = rng.choice(J.size, size=ns, p=J.ravel()); cA, cB = np.unravel_index(flat, J.shape)
        else:
            # Multinomial (SMOKE TESTS ONLY): a Poisson tail at 38 per voxel reaches the 64-slot cap over 16k voxels,
            # where the reference's hard-sphere start (Fano ~0.6) does not. Over-cap voxels are redrawn and counted.
            cA = np.bincount(rng.integers(0, ns, size=NA), minlength=ns)
            cB = np.bincount(rng.integers(0, ns, size=Bspec[0].count), minlength=ns)
            redrawn = 0
            while (cA + cB).max() > max_occ:
                bad = np.flatnonzero(cA + cB > max_occ); redrawn += len(bad)
                cA[bad] = rng.poisson(cA.mean(), len(bad)); cB[bad] = rng.poisson(cB.mean(), len(bad))
            if redrawn: print(f'[ternary_relax] multinomial start: {redrawn} over-cap voxel(s) redrawn (smoke path; the seeded start cannot exceed the reference)', flush=True)
        if (cA + cB).max() > max_occ:
            raise RuntimeError('stripe start needs %d slots but MAX_OCC is %d' % ((cA + cB).max(), max_occ))
        for s, a, b in zip(sites, cA, cB):
            if a: arr[s[0], s[1], s[2], :a] = 1
            if b: arr[s[0], s[1], s[2], a:a + b] = 3
        lat[...] = arr
        return float(cA.mean()), float(cB.mean()), int((cA + cB).max())


class Probe(lm.DriftMpdRdmeSolver):
    """Re-arms the opt-ins runSolver's init resets, then records A/B occupancy and every C's COM at each hook."""
    def setup(self):
        self.n = 0; self.t = []; self.occA = []; self.occB = []; self.posC = []; self.xaxis = None
    def hookSimulation(self, t, lattice):
        self.n += 1
        if self.n == 1:
            if excl:
                self.setStructureReciprocalVex(True, BATH_ID)
                if A.arm == 'model': self.setStructureFreeVolume(1)
            else:
                self.setStructureCollisionalFriction(0)          # no bath lattice published: legal
            set_order(self)                                     # re-armed with the other opt-ins, idempotent
            return 0
        v = np.asarray(lattice.getParticleLatticeView())
        oA = (v == 1).sum(axis=(0, v.ndim - 1)); oB = (v == 3).sum(axis=(0, v.ndim - 1))
        if self.xaxis is None:                                  # which view axis is x: the one carrying the stripe
            prof = [oA.mean(axis=tuple(a for a in range(3) if a != ax)) for ax in range(3)]
            contrast = [p.max() - p.min() for p in prof]; self.xaxis = int(np.argmax(contrast))
        # The view's spatial axes come out (z, y, x) on this build (stripe axis detected = 2); reverse to (x, y, z) so
        # the raveled index is (ix*N + iy)*N + iz, the ReaDDy binning, and matches the COM order (x, y, z). A view that
        # already carried x first is left alone.
        if self.xaxis == 2: oA = oA.transpose(2, 1, 0); oB = oB.transpose(2, 1, 0)
        elif self.xaxis == 1: oA = np.moveaxis(oA, 1, 0); oB = np.moveaxis(oB, 1, 0)
        o3 = np.zeros(3); coms = np.zeros((len(centers), 3))
        for i in range(len(centers)):
            self.getStructureCOM(i, o3); coms[i] = o3
        self.t.append(float(t)); self.occA.append(oA.ravel().astype(np.int16)); self.occB.append(oB.ravel().astype(np.int16))
        self.posC.append(coms)
        return 0


def configure(sol):
    xyz = XYZ
    sol.setPotentialBasis(np.zeros(xyz, np.float32), np.zeros(xyz, np.float32))
    sol.setStructureHandler(1, SIGMA_FILL, V); sol.configureStructureLattice(NX, NY, NZ)
    diam = bsp.diameters(Bspec, sig['A'], SIGMA_FILL)              # [A, FILL, B] = types 1, 2, 3
    sol.setSpeciesHardSphereDiameters(diam, V)
    for t in range(1, len(diam) + 1): sol.setSpeciesCoupling(t - 1, 0.0, 0.0)
    if excl: sol.setHardCoreExclusion(LAM_HC, lm.HCForm_WHITE_BEAR if hasattr(lm, 'HCForm_WHITE_BEAR') else 2)
    l1 = np.zeros(xyz, np.int32); l2 = np.full(xyz, DIR_NONE, np.int32); l3 = np.zeros(xyz, np.int32)
    coms, flags, nsub = [], [], []
    for i, (x, y, z) in enumerate(centers):
        v = vox(x, y, z); l1[v] = i + 1; l3[v] = (ROLE_BODY << 4) | 1
        coms += [float(x), float(y), float(z)]; nsub.append(1); flags.append(FLAG_ACTIVE | FLAG_INERT | (RIBO_ID << 8))
    sol.setStructureOverlay(l1, l2, l3)
    sol.setStructureRegistry(len(centers), np.array(nsub, np.int32), np.array(coms, np.float64), np.array(flags, np.int32))
    for i, (x, y, z) in enumerate(centers): sol.setStructureTips(i, vox(x, y, z), vox(x, y, z), -1)
    sol.setStructureSizeDiffusion(D_MODEL_KIRKWOOD, D1_UM2 * 1e6, TAU_S, NM)
    sol.setStructureHopSpeciesScale(RIBO_ID, D_UM2['C'] / D1_UM2)  # C at D_C = D_A sigma_A / sigma_C before friction
    if excl:
        sol.setStructureSmearStencil(rb.fmt_stencil(sig['C'], NM).reshape(-1).astype(np.float32))
        sol.setStructureSmearDnaBase(np.zeros(4 * xyz, np.float64)); sol.armStructureSmearRefresh(RIBO_ID)
        sol.setStructureSmearIncremental(512); sol.setStructureReciprocalVex(True, BATH_ID)
        if A.arm == 'model': sol.setStructureFreeVolume(1)
    else:
        sol.setStructureCollisionalFriction(0)
    sol.setHopIntegration(1)
    if not A.fixed_tau: sol.setAdaptiveTau(1, 0.2, 0.05)
    set_order(sol)
    sol.validateStability()


def set_order(sol):
    """The step-order switches of the rand-split build; a build without them stops rather than running fixed."""
    if A.order == 'perm' or A.smear_after:
        if not hasattr(sol, 'setStepOrderRandomize'): sys.exit('--order perm / --smear-after need the rand-split build')
    if A.order == 'perm': sol.setStepOrderRandomize(1)
    if A.smear_after: sol.setStructureSmearAfterMoves(1)


def main():
    t0 = time.time()
    box = micron(NX * NM * 1e-3, NY * NM * 1e-3, NZ * NM * 1e-3)
    sim = RDME.RDMESimulation(dimensions=box, spacing=nm(NM))
    sim.defineSpecies(['A']); sim.modifyRegion('default').setDefaultDiffusionRate(D_A_UM2 * 1e-12)
    bsp.define_in_sim(sim, Bspec, NX, NY, NZ, NM)                    # 'FILL' (2) + 'B2' (3), B particles added (overwritten below)
    sim.addParticles(species='A', region='default', count=NA)
    scene.finish_sim(sim, A.seed, TAU_S, t_max, per_frame)          # seed, dt, hook every per_frame steps, no file writes
    joint = np.load(A.seed_hist) if A.seed_hist and not A.uniform else None
    print(f"[ternary_relax] arm={A.arm} dims={NX}x{NY}x{NZ} h={NM:g} sig={sig} eta=({A.eta_a},{A.eta_b}) NA={NA} NB={Bspec[0].count} "
          f"C={len(centers)} D_um2={ {k: round(v, 4) for k, v in D_UM2.items()} } tau_max={A.tau_max} -> {n_steps} nominal steps, hook every {per_frame}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ternary.lm'); sim.save(path)
        mA, mB, mx = stripe_lattice(path, joint)
        print(f"[ternary_relax] stripe start: filled-voxel mean A {mA:.2f} B {mB:.2f}, max total {mx} (cap 64), source {'joint hist' if joint is not None else 'multinomial'}", flush=True)
        sol = Probe(); sol.setup(); configure(sol)
        lm.runSolver(path, 1, sol, [A.gpu], 0)
    err = np.zeros(4, np.int32); sol.getStructureError(err)
    over = -1.0
    if A.arm == 'model':
        o = np.zeros(1, np.float64); sol.getStructureFreeVolumeStat(o); over = float(o[0])
    tau_scale = float(sol.getTauScale())
    t = np.asarray(sol.t); tau = t * D_A_UM2 * 1e6 / NM ** 2
    meta = dict(arm=A.arm, seed=A.seed, N=N, dims=[NX, NY, NZ], h_nm=NM, sigma=sig, D_um2=D_UM2, eta_a=A.eta_a, eta_b=A.eta_b, NA=NA,
                NB=int(Bspec[0].count), n_c=len(centers), centers=centers, period=PERIOD, duty=DUTY, tau_max=A.tau_max,
                n_steps_nominal=n_steps, per_frame=per_frame, tau_scale=tau_scale, tau_us=A.tau_us, fixed_tau=A.fixed_tau, order=A.order, smear_after=A.smear_after, struct_err=int(err[0]),
                free_vol_overlap_reads=over, view_x_axis=sol.xaxis, seed_hist=A.seed_hist, wall_s=time.time() - t0)
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    np.savez_compressed(A.out, meta=json.dumps(meta), tau=tau, occA=np.stack(sol.occA), occB=np.stack(sol.occB),
                        posC=np.stack(sol.posC))
    a = sol.occA[-1].astype(float); b = sol.occB[-1].astype(float)
    print(f"[ternary_relax] -> {A.out}: {len(tau)} frames, tau {tau[0]:.3g}..{tau[-1]:.3g}, tau_scale {tau_scale:.3f}, "
          f"struct_err {int(err[0])}, overlap reads {over:.0f}, final Fano A {a.var()/a.mean():.3f} B {b.var()/b.mean():.3f}, {time.time()-t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
