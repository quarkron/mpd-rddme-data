"""Semiflexible chains, dry or in a byte-lattice bath: the filament driver of the ERA structure loop.

One fixed-length chain of N beads on the structure handler, chemistry off (no formation channel, no tip driver), moved
by the interior conformational kernel and, when requested, the collective rigid hop and the rigid rotation, all through
the production path (launchStructurePasses inside the RDME timestep), so the kernels timed here are the kernels a full
simulation runs. The geometry and seeding come from fixed_chain_harness.py (edge arm by default: rest length root two
with a stiff stretch term selects the twelve edge diagonals, so the bending stiffness is the only conformational
parameter).

With --eta > 0 the lattice carries a bath of 3 nm particles at that packing fraction. The bath feels the chain through
the reciprocal fill (each bead counts as one particle of the fill diameter in xiAt). The chain feels the bath through
the engine's chain friction (default since 2026-09-10, polymer notes Appendix B): the mobile bath is OUT of the move
energy and every move is thinned by m = 1/(1 + c <phi>_shell), <phi> the bath fraction in the 26 voxels around each
displaced bead, mean over the beads a move displaces. The coefficient c = (1/2) D1/(D1+D_B) (1+lambda)^3/lambda^3
is derived by the engine from the FILL species (diameter --sigma-bead, diffusion --d1-um2) and the bath species
(3 nm, --d-bath); friction_closure.coefficient reproduces it here for the record. Charging the chain the White Bear
insertion work against the bath instead (--legacy-bath-energy) is the configuration that FREEZES it at eta 0.15
(one conformation in 100 frames): kept as the demonstration of the cage, logged at WARNING by the engine.
Owner choice for the transport figure: D_B = D1 = 0.3 um^2/s, which needs tau = 20 us (default since 2026-09-10): the
drift-RDME bath transport runs away above q = D_B tau/h^2 ~ 0.08 at eta 0.15 whatever the chain does (see --d-bath).
ReaDDy must run at the same D ratio.

Frames hold the chain's centre of mass (registry) and the ordered backbone positions (overlay readout), so the loop can
hash the trajectory and compute the Tier-1 chain observables (bond lengths, Rouse modes, end-to-end distance) offline.

  python3 chain_bath.py --n 32 --kappa 2 --eta 0 --bench 4000 --frame-steps 40 --out C32_dry.npz
  python3 chain_bath.py --n 32 --kappa 2 --eta 0.15 --nx 64 --ny 64 --nz 64 --bench 2000 --frame-steps 20 --out C32_bath.npz

Env overrides: STRUCT_GPU, STRUCT_PROFILE, LM_VERBOSITY. Shared machinery: bath_scene.py.
"""
import argparse
import os
import sys

import numpy as np
import lm
from pyLM import RDME
from pyLM.units import nm, micron

import bath_scene as scene
import chain_geometry as G
import fcc_exact as FX
import fixed_chain_harness as H
import friction_closure as FC

NM_PER_VOXEL = 10.0
TAU_S = 20e-6                          # default since 2026-09-10 (--tau-us): the bath transport runs away above q = D_B tau/h^2 ~ 0.08 at eta 0.15
SIGMA_BATH = 3.0                       # bath hard-sphere diameter (nm), as in stamp_msd.py
BATH_ID, FILL_ID = 1, 2                # byte-lattice types: 1 = bath 'B', 2 = the bead's fill diameter row 'FILL' (no particles)
LAM_HC = float(os.environ.get('CHAIN_BATH_LAM_HC', '1.0'))   # 0 = pure diffusion (probe of the bath transport without the White-Bear drift)


def bath_count(eta, nx, ny, nz):
    return int(round(eta * nx * ny * nz * NM_PER_VOXEL ** 3 / ((np.pi / 6.0) * SIGMA_BATH ** 3)))


def build_sim(args, t_end, hook_steps, lattice_write_interval):
    nx, ny, nz = args.nx, args.ny, args.nz
    n_bath = bath_count(args.eta, nx, ny, nz)
    box_um = (nx * NM_PER_VOXEL * 1e-3, ny * NM_PER_VOXEL * 1e-3, nz * NM_PER_VOXEL * 1e-3)
    sim = RDME.RDMESimulation(dimensions=micron(*box_um), spacing=nm(NM_PER_VOXEL))
    sim.defineSpecies(['B', 'FILL'])
    sim.modifyRegion('default').setDefaultDiffusionRate(args.d_bath * 1e-12)
    sim.modifyRegion('default').setDiffusionRate('FILL', args.d1_um2 * 1e-12)   # no FILL particles exist; its D is the subunit D1 the closure reads
    if n_bath > 0:
        sim.addParticles(species='B', region='default', count=n_bath)
    scene.finish_sim(sim, args.seed, TAU_S, t_end, hook_steps, scene.NEVER_STEPS, lattice_write_interval)
    print(f"[chain_bath] box {nx}x{ny}x{nz}, chain n={args.n} kappa={args.kappa} arm={args.arm} seed_kind={args.seed_kind}; "
          f"bath eta={args.eta}: {n_bath} particles of {SIGMA_BATH} nm (mean occupancy {n_bath/(nx*ny*nz):.2f}); "
          f"t_end={t_end:g}s dt={TAU_S*1e6:.0f}us hook every {hook_steps} steps seed={args.seed}")
    return sim, n_bath


def configure(sol, args, rng):
    nx, ny, nz = args.nx, args.ny, args.nz
    xyz = nx * ny * nz
    cfg = H.ARMS[args.arm]
    sol.setPotentialBasis(np.zeros(xyz, np.float32), np.zeros(xyz, np.float32))   # no external potential
    sol.setSpeciesCoupling(BATH_ID - 1, 0.0, 0.0)
    sol.setSpeciesCoupling(FILL_ID - 1, 0.0, 0.0)
    sol.setStructureHandler(1, args.sigma_bead, NM_PER_VOXEL ** 3)
    sol.configureStructureLattice(nx, ny, nz)
    sol.setSpeciesHardSphereDiameters(np.array([SIGMA_BATH, args.sigma_bead], np.float64), NM_PER_VOXEL ** 3)
    sol.setHardCoreExclusion(LAM_HC, lm.HCForm_WHITE_BEAR if hasattr(lm, 'HCForm_WHITE_BEAR') else 2)
    if args.hop_integration:
        sol.setHopIntegration(int(args.hop_integration))

    if args.chains > 1 and args.seed_kind != 'exact':
        raise ValueError("more than one chain is only placed from the exact sampler (--seed-kind exact)")
    if args.seed_kind == 'straight':
        chains = [H.straight_walk(args.n, nx, ny, nz, cfg['orbit'])]
    elif args.seed_kind == 'coil':
        chains = [H.self_avoiding_walk(args.n, nx, ny, nz, rng, cfg['orbit'])]
    elif args.seed_kind == 'exact':
        if cfg['orbit'] != 'edge':
            raise ValueError("the exact seed is the FCC closed-form sampler; arm '%s' has none" % args.arm)
        chains = FX.seed_many(args.chains, args.n, args.kappa, nx, ny, nz, rng)
    else:
        raise ValueError("unknown seed kind %r" % args.seed_kind)
    H.seed_chains(sol, chains, nx, ny, nz, species=FILL_ID)   # FILL_ID carries the bead's hard-sphere diameter
    pos = chains[0][0]

    # The interior attempt budget is per STEP; keep the attempt RATE at the 50 us convention (n-1 per 50 us) when tau
    # changes, or a shorter step would speed the chain's conformational clock against the bath and the rigid hop.
    att = int(args.attempts) if args.attempts else max(1, int(round((args.n - 1) * TAU_S / 50e-6)))
    sol.setStructureLaunchParams(att, float(args.kappa))
    sol.setStructureStretch(cfg['k_stretch'], cfg['rest_len'])
    sol.setStructureInteriorMoveMask(int(args.move_mask))
    arm_topology(sol, first=True)
    d1_nm2 = float(args.d1_um2) * 1e6                    # the setters take nm^2/s with the voxel edge in nm
    if args.rigid_translation:
        sol.setStructureSizeDiffusion(0, d1_nm2, TAU_S, NM_PER_VOXEL)   # Rouse D(N) = D1/N
    if args.rigid_rotation:
        sol.setStructureRotationModel(1, d1_nm2, TAU_S, NM_PER_VOXEL)   # Tirado rod
    # Armed here as well as in the rearm hook: runSolver's init resets it, so this call is the one
    # that covers everything before the first hook, and the rearm covers everything after.
    if args.eta > 0:
        sol.setStructureReciprocalVex(True, FILL_ID)
        arm_chain_friction(sol, args)
    print(f"[chain_bath] attempts/step={att} k_stretch={cfg['k_stretch']} rest_len={cfg['rest_len']:.4f} "
          f"rigid_translation={args.rigid_translation} rigid_rotation={args.rigid_rotation} "
          f"d1={args.d1_um2} um^2/s seed={args.seed_kind} move_mask={args.move_mask}")
    return pos


def make_rearm(args):
    """Re-arm the bath coupling after runSolver's init drops it.

    Armed only when there IS a bath. Arming it at eta = 0 was tried and reverted: with reciprocal
    VEX on, structVexLatticeG points at the live lattice and every move pays the White-Bear insert
    against it, which cost 2.8x on the dry chain suite points for no physics at all — there is
    nothing for a chain to feel in an empty box.

    What made the original bug silent was not this gate. It was that the chain carried no species
    tag, so structVexInsertDelta was skipped even WITH the bath present; eta 0 and eta 0.15 returned
    bit-identical trajectories. That failure mode is now impossible: fixed_chain_harness._flags
    refuses species 0. This is armed in configure() as well as here, because runSolver's init resets
    it and the two calls between them cover the whole run.
    """
    def rearm(sol):
        if args.eta > 0:
            sol.setStructureReciprocalVex(True, FILL_ID)   # bath feels the beads; the beads feel the live field
            arm_chain_friction(sol, args)
        arm_topology(sol, first=False)
    return rearm


def topology_flags():
    """The strand-passage rule, the interior move mask and the hop orbit, from the environment. The ERA suite runs its
    points through env flags only (`ENVX`), so these are the knobs the crossing change's cost measurement needs:
    STRUCT_BOND_CROSSING=1 arms the rule, STRUCT_INTERIOR_MASK=0x1B selects the topology-conserving set (end-bond,
    full kink, edge translation), STRUCT_HOP_ORBIT=1 draws the collective hop from the twelve edge vectors."""
    def geti(name):
        v = os.environ.get(name)
        return None if v in (None, '') else int(v, 0)
    return geti('STRUCT_BOND_CROSSING'), geti('STRUCT_INTERIOR_MASK'), geti('STRUCT_HOP_ORBIT')


def arm_topology(sol, first):
    """Applied at configure and again in the rearm hook: the handler's enable block resets the mask to 0x7 and the
    orbit to the face set. The rule itself survives the init and is armed once, because arming zeroes its counters."""
    rule, mask, orbit = topology_flags()
    sweep = os.environ.get('STRUCT_INTERIOR_SWEEP')
    if sweep not in (None, ''):
        sol.setStructureInteriorSweep(int(sweep))          # 1 = one attempt per subunit per step in coloured sweeps
    if mask is not None:
        sol.setStructureInteriorMoveMask(mask)
    if orbit is not None:
        sol.setStructureHopOrbit(orbit)
    if first and rule is not None:
        sol.setStructureBondCrossing(rule)


def arm_chain_friction(sol, args):
    """Chain friction is the engine default; this applies the driver's overrides (legacy energy, explicit c)."""
    if args.legacy_bath_energy:
        sol.setStructureChainFriction(0)                     # the cage, on purpose (engine logs WARNING)
    elif args.chain_c is not None:
        sol.setStructureChainFrictionCoefficient(BATH_ID, float(args.chain_c))


def expected_chain_c(args):
    """The coefficient the engine derives: fill species (sigma_bead, D1) against the bath (3 nm, D_B)."""
    return FC.coefficient(args.sigma_bead, SIGMA_BATH, args.d1_um2, args.d_bath)


def distinct_conformations(beads):
    """Number of distinct conformations over the frames, translation removed (the freeze diagnostic: report it beside
    every crowded statistic)."""
    b = np.asarray(beads, np.int64)
    if b.ndim == 4: b = b[:, 0]
    rel = b - b[:, :1, :]
    return len({r.tobytes() for r in rel})


def backbone_reader(nx, ny, nz, n, k_chains=1):
    def read(sol):
        got = G.read_chains(sol, nx, ny, nz)
        out = []
        for cid in range(1, k_chains + 1):
            ch = got.get(cid)
            if ch is None or not ch['ok']:
                raise RuntimeError("backbone readout failed for complex %d: %s"
                                   % (cid, (ch or {}).get('reason', 'walk broke')))
            if ch['n'] != n:
                raise RuntimeError(f"complex {cid} length changed from {n} to {ch['n']}")
            out.append(np.asarray(ch['pos'], np.int64))
        return np.stack(out) if k_chains > 1 else out[0]
    return read


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32, help='beads')
    ap.add_argument('--kappa', type=float, default=2.0, help='bending stiffness (kT)')
    ap.add_argument('--arm', default='edge', choices=sorted(H.ARMS))
    ap.add_argument('--seed-kind', dest='seed_kind', default='coil',
                    choices=('coil', 'straight', 'exact'),
                    help="'exact' draws from the closed-form FCC distribution (fcc_exact.seed), so an "
                         "equilibrium measurement needs a stationarity check rather than a burn-in")
    ap.add_argument('--move-mask', dest='move_mask', type=int, default=H.MOVE_ALL,
                    help='interior move classes: 1 end-bond, 2 kink, 4 crankshaft, 7 all, 0 none. '
                         '0 isolates the rigid hop as the only channel that moves the centre of mass')
    ap.add_argument('--attempts', type=int, default=0, help='interior attempts per step (0 = (n - 1) * tau / 50 us, i.e. the 50 us convention held at any tau)')
    ap.add_argument('--sigma-bead', dest='sigma_bead', type=float, default=10.0, help='bead diameter (nm), at most the voxel edge')
    ap.add_argument('--d1-um2', dest='d1_um2', type=float, default=0.3, help='single-bead D for the rigid moves (um^2/s)')
    ap.add_argument('--no-rigid-translation', dest='rigid_translation', action='store_false')
    ap.add_argument('--no-rigid-rotation', dest='rigid_rotation', action='store_false')
    ap.add_argument('--eta', type=float, default=0.0, help='bath packing fraction (0 = dry chain)')
    ap.add_argument('--d-bath', dest='d_bath', type=float, default=0.3,
                    help='bath D (um^2/s); default = D1 (owner choice 2026-09-10; the closure carries D1/(D1+D_B), so ReaDDy must run '
                         'at the same ratio). Needs --tau-us small enough: the drift-RDME bath transport runs away (geometric particle '
                         'overflow, chain static or not, LINEAR or EXACT hop) above q = D_B tau/h^2 ~ 0.08 at eta 0.15, consistent with a '
                         'collective-diffusion CFL 6 q (1+8 eta) < 1; tau 20 us (q 0.06) ran 40000 steps x 2 seeds with zero overflows, '
                         'tau 25 (q 0.075) too but on the edge, tau >= 30 (q >= 0.09) overflows within 10 steps')
    ap.add_argument('--legacy-bath-energy', dest='legacy_bath_energy', action='store_true',
                    help='charge the chain the White Bear insertion work against the mobile bath (the configuration that freezes it); demonstration only')
    ap.add_argument('--chain-c', dest='chain_c', type=float, default=None,
                    help='explicit chain-friction coefficient for the bath species (overrides the engine closure; 0 = bath-blind control)')
    ap.add_argument('--tau-us', dest='tau_us', type=float, default=20.0,
                    help='RDME time step (us); default 20 (was 50 until 2026-09-10). The bath hop probability per direction is '
                         'q = D_B tau / h^2 and the bath transport runs away above q ~ 0.08 at eta 0.15 (see --d-bath). The interior '
                         'attempt budget scales with tau so the chain\'s clock is unchanged; the rigid moves read tau directly. '
                         'Dry results at the old convention: --tau-us 50')
    ap.add_argument('--hop-integration', dest='hop_integration', type=int, default=0, choices=(0, 1),
                    help='bath hop integration: 0 = LINEAR (legacy, exact free D, needs q*B < 1/2), 1 = EXACT (bounded 1-exp(-W tau); '
                         'free D is low by (1-exp(-2q))/(2q))')
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--ny', type=int, default=32)
    ap.add_argument('--nz', type=int, default=32)
    ap.add_argument('--t-end', dest='t_end', type=float, default=0.1)
    ap.add_argument('--frame-dt', dest='frame_dt', type=float, default=0.001)
    ap.add_argument('--chains', type=int, default=1,
                    help='number of independent chains in one box. The structure kernels never see the '
                         'simulation seed but do mix the complex id into the move hash, so K chains in '
                         'one run are K independent streams where K runs are one stream repeated')
    ap.add_argument('--out', default='chain_bath.npz')
    scene.add_scene_args(ap)
    return ap.parse_args()


def main():
    global TAU_S
    args = parse_args()
    TAU_S = float(args.tau_us) * 1e-6
    if LAM_HC != 1.0:
        print(f"[chain_bath] PROBE: hard-core coupling lambda = {LAM_HC} (CHAIN_BATH_LAM_HC)")
    if args.sigma_bead > NM_PER_VOXEL:
        sys.exit(f"--sigma-bead {args.sigma_bead} exceeds the voxel edge {NM_PER_VOXEL} nm (setStructureHandler requires sigma <= h)")
    q = args.d_bath * 1e6 * TAU_S / NM_PER_VOXEL ** 2
    if args.eta > 0 and 6.0 * q > 0.98:
        sys.exit(f"CFL: bath 6*D*dt/h^2 = {6*q:.3f} > 0.98; lower --d-bath")
    if args.bench:
        frame_steps = args.frame_steps or max(1, int(round(args.frame_dt / TAU_S)))
        t_end, hook_steps = scene.bench_plan(args.bench, frame_steps, TAU_S)
        rec = dict(frame_steps=hook_steps)
        print(f"[chain_bath] BENCH: {args.bench} steps, one frame every {hook_steps} steps ({args.bench // hook_steps} frames)")
    else:
        t_end = args.t_end
        hook_steps = max(1, int(round(args.frame_dt / TAU_S)))
        rec = dict(frame_dt=args.frame_dt)

    gpu = int(os.environ.get('STRUCT_GPU', '0'))
    lm.setVerbosityLevel(int(os.environ.get('LM_VERBOSITY', '1')))
    rng = np.random.default_rng(args.seed)
    sim, n_bath = build_sim(args, t_end, hook_steps, scene.lattice_write_steps(t_end, TAU_S, args.lattice_frames))
    sol = scene.BathRecorder()
    pos0 = configure(sol, args, rng)
    sol.configure_recording(args.chains, (args.nx, args.ny, args.nz), rearm=make_rearm(args),
                            extra=backbone_reader(args.nx, args.ny, args.nz, args.n, args.chains),
                            progress_every=0, **rec)
    wall = scene.run_scene(sim, sol, gpu, args.reuse_lm, args.save_init_lm, args.keep_lm, tag='chain_bath')

    err = np.zeros(4, np.int32)
    sol.getStructureError(err)
    if os.environ.get('STRUCT_PROFILE'):
        sol.printStructureProfile()
    times, coms, backbones = sol.frames()
    if coms.size == 0:
        sys.exit("no frames recorded; check the hook interval and simulation time")
    beads = np.asarray(backbones, np.int64)   # (F, n, 3) or (F, K, n, 3), unwrapped integer voxels
    ree = np.linalg.norm(beads[..., -1, :] - beads[..., 0, :], axis=-1) * NM_PER_VOXEL
    bl = np.linalg.norm(np.diff(beads, axis=-2), axis=-1)   # bond lengths in voxels
    ndist = distinct_conformations(beads)
    fric = dict(friction=bool(args.eta > 0 and not args.legacy_bath_energy), chain_c=float('nan'), chain_c_expected=expected_chain_c(args),
                friction_decisions=0.0, friction_mean_m=float('nan'), friction_shell_count=float('nan'),
                m_expected=FC.mobility(args.eta, expected_chain_c(args)))
    if fric['friction']:
        fric['chain_c'] = float(args.chain_c) if args.chain_c is not None else fric['chain_c_expected']
        fs = np.zeros(3, np.float64)
        sol.getStructureChainFrictionStat(fs)
        fric.update(friction_decisions=float(fs[0]), friction_mean_m=float(fs[1]), friction_shell_count=float(fs[2]))
    print(f"[chain_bath] frames={len(times)} t_end={times[-1]:.4f}s wall={wall:.1f}s struct_err={int(err[0])}  "
          f"<R_ee>={ree.mean():.1f} nm  <bond>={bl.mean():.4f} vox (edge arm: {np.sqrt(2):.4f})  "
          f"chains={args.chains}  COM drift={np.linalg.norm(coms[-1, 0] - coms[0, 0]) * NM_PER_VOXEL:.1f} nm  "
          f"distinct conformations={ndist}/{len(times)}")
    if topology_flags()[0]:
        print(f"[chain_bath] strand-passage rule armed (mask {topology_flags()[1]}, hop orbit {topology_flags()[2]}): "
              f"refusals crossing={sol.getStructureCrossBlocked()} chord={sol.getStructureChordBlocked()} "
              f"policy={sol.getStructurePolicyRefused()}")
    if fric['friction']:
        print(f"[chain_bath] chain friction: c={fric['chain_c']:.3f} (closure {fric['chain_c_expected']:.3f}), decisions={int(fric['friction_decisions'])}, "
              f"mean m={fric['friction_mean_m']:.4f} (1/(1+c eta)={fric['m_expected']:.4f}), shell voxels={fric['friction_shell_count']:.1f}")
    np.savez_compressed(args.out, times=times, coms_vox=coms, beads_vox=beads, pos0_vox=np.asarray(pos0, np.int64),
                        n=args.n, chains=args.chains, kappa=args.kappa, arm=args.arm, seed_kind=args.seed_kind, eta=args.eta, n_bath=n_bath,
                        nx=args.nx, ny=args.ny, nz=args.nz, nm_per_voxel=NM_PER_VOXEL, tau_s=TAU_S, sigma_bath=SIGMA_BATH,
                        sigma_bead=args.sigma_bead, d1_um2=args.d1_um2, d_bath_um2=args.d_bath,
                        rigid_translation=bool(args.rigid_translation), rigid_rotation=bool(args.rigid_rotation),
                        reciprocal=bool(args.eta > 0), legacy_bath_energy=bool(args.legacy_bath_energy), distinct_conformations=ndist,
                        seed=args.seed, bench_steps=int(args.bench), struct_err=int(err[0]), wall_s=wall, **fric)
    print(f"[chain_bath] wrote {args.out}")
    if int(err[0]) != 0:
        sys.exit(f"struct_err={int(err[0])}")


if __name__ == '__main__':
    main()
