#!/usr/bin/env python3
"""Polymer growth and dynamics in a crowder density gradient (dense core, sparse periphery), FCC chains.

THE QUESTION. Where does a polymer nucleate and grow when monomer transport, monomer partitioning and the polymer's own
friction all vary in space, and does it drift once formed? The rate-factor coupling (polymer notes, Appendix B) promises
one clean control: a friction gradient alone produces NO drift, so any radial motion of a polymer must come from a real
free-energy gradient (monomers partitioned out of the core by excluded volume, and nothing else in this model).

THE MODEL. Monomers M (byte lattice, sigma = h = 10 nm, D 0.3 um^2/s) dispersed uniformly at t = 0; crowders C held in a
radial well psi(r) = A min(r, r_edge)^2 / r_edge^2 with A = ln(phi_core/phi_edge), so their equilibrium profile falls from
phi_core at the centre to phi_edge at r_edge and stays flat beyond (periodic box, no discontinuity); monomers are not
coupled to the well (they partition through excluded volume only). Chains are FCC (one subunit per voxel, edge orbit:
rest length sqrt 2, stretch 400) with bending kappa. Chemistry: nucleation M + M -> dimer, polymerisation at BOTH tips
(the unified tip kernel adds at either end of every POLYMERIZE channel), depolymerisation at both tips (the - tip retract
and the reversible + tip shed), and fragmentation anywhere (SEVER, per subunit); poly faster than depoly, no annealing.
The chain friction (default) puts the mobile bath into the chain's move RATE; the formation excess is the monomer ->
subunit SWAP (default), so growth is proportional to the local monomer count and crowding enters once, through
partitioning.

CROWDERS, two representations (--crowder-mode).
  smear (default): 20 nm bodies carried as depositing structure complexes with the fundamental-measure SMEAR stencil (the
     ribosome machinery): their exclusion footprint reaches the neighbouring voxels, so monomers and subunits are pushed
     out of the dense core with the continuum strength, two subunits sharing a crowder's footprint feel DEPLETION, and the
     chain feels the crowders as an energy (setStructureSmearExtAll). Bodies move on the sub-voxel hop with the
     collisional friction against the monomer bath, and the well acts on the body species. Bath statics next to a body
     are the one-sided ones unless --free-volume arms the free-volume functional.
  point: 10 nm crowders as byte-lattice particles (lambda = 1). Exclusion is booked in the crowder's own voxel only, so
     equal-size excluded volume is under-represented ~3x and depletion is absent; the chain feels them only through the
     rate factor (no static term in a gradient). Kept as the mean-field reference. A 3 nm bath (--sigma-crowd 3) self-traps
     a 10 nm monomer in the hop kernel until the rate factor is extended to byte-lattice species.

CONTROL. --no-chemistry --seed-chains K places K exact-seeded FCC chains and no monomers: their centre-of-mass radial
distribution must stay uniform per volume in the gradient (the zero-drift test of the symmetric rate factor).

OUTPUT (npz): times; per frame the number of chains, total bound subunits, free monomers, length histogram, radial
histograms (10 nm shells from the box centre) of M, C, chain centres of mass and bound subunits, nucleation count so far;
the final backbones; the friction diagnostic; every parameter. Frames every --frame-dt.

  python3 gradient_growth.py --t-end 5 --out /raid/racda/lmrddme_polymer/gradient/G_run1.npz
  python3 gradient_growth.py --no-chemistry --seed-chains 16 --t-end 2 --out .../G_control.npz
Run inside the lm2.6 container (PYTHONPATH with testing/engine). STRUCT_GPU selects the device.
"""
import argparse
import os
import sys
import time

import numpy as np
import lm
from pyLM import RDME
from pyLM.units import nm, micron, microsecond

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Lattice_Microbes_2.6', 'testing', 'engine'))
import chain_geometry as G                                     # noqa: E402  (read_chains: overlay -> ordered backbones)
import fixed_chain_harness as H                                # noqa: E402  (ARMS['edge'], seed_chains)
import fcc_exact as FX                                         # noqa: E402  (exact FCC seeds for the control)
import friction_closure as FC                                  # noqa: E402
import bath_scene as scene                                     # noqa: E402  (finish_sim: seed, clock, hook cadence in STEPS)
import ribosome_blob as rb                                     # noqa: E402  (fmt_stencil: the 20 nm body's fundamental-measure footprint)
import gradient_bench as GB                                    # noqa: E402  (the structure loop's bench mode and --state-from)

R_ID = 2                                                       # body species tag (smear mode); its byte row is a placeholder diameter
SIGMA_BODY_ROW = 5.0                                           # placeholder byte diameter for the body species (the smear carries the 20 nm)
FLAG_ACTIVE, FLAG_RIGID, FLAG_INERT, DIR_NONE, ROLE_BODY = 1, 0x4, 0x10, 255, 0

H_NM = 10.0
M_ID, C_ID = 1, 2
RXN_NUCLEATE, RXN_POLYMERIZE, RXN_DEPOLYMERIZE, RXN_SEVER = 0, 1, 2, 3
WB = lm.HCForm_WHITE_BEAR if hasattr(lm, 'HCForm_WHITE_BEAR') else 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=64, help='cubic box edge in voxels (h = 10 nm)')
    ap.add_argument('--tau-us', dest='tau_us', type=float, default=20.0)
    ap.add_argument('--t-end', dest='t_end', type=float, default=5.0)
    ap.add_argument('--frame-dt', dest='frame_dt', type=float, default=0.01)
    ap.add_argument('--sigma', type=float, default=10.0, help='monomer = subunit diameter (nm); 10 = one subunit per voxel (FCC)')
    ap.add_argument('--crowder-mode', dest='crowder_mode', default='smear', choices=('smear', 'point'))
    ap.add_argument('--sigma-crowd', dest='sigma_crowd', type=float, default=None, help='crowder diameter (nm); default 20 (smear) / 10 (point)')
    ap.add_argument('--d-mono', dest='d_mono', type=float, default=0.3, help='um^2/s')
    ap.add_argument('--no-vex', dest='no_vex', action='store_true',
                    help='the control with volume exclusion OFF: the White-Bear prefactor set to zero, the reciprocal '
                         'term, the per-subunit smear term and both hard cores disabled. What remains is the lattice '
                         'itself -- a voxel still holds one subunit, and a body still occupies its own voxel -- so this '
                         'is not a point-particle limit in the continuum sense, and in POINT crowder mode, where the '
                         'crowders live on the byte lattice instead of the overlay, nothing of them is left at all.')
    ap.add_argument('--save-beads', dest='save_beads', action='store_true',
                    help='store every chain backbone at every frame: what a Rouse-mode transform needs, and small '
                         '(8 chains x 11 subunits x 2000 frames is 4 MB)')
    ap.add_argument('--snap-frames', dest='snap_frames', type=int, default=0,
                    help='store the body and chain voxel indices at this many evenly spaced frames (0 = none): the spatial '
                         'snapshots a figure needs, at a few kB per frame instead of a lattice dump')
    ap.add_argument('--bath-eta', dest='bath_eta', type=float, default=0.0,
                    help='volume fraction of a SMALL byte-lattice bath carried alongside the 20 nm bodies (smear mode): the ternary '
                         'scene, where the bath also induces depletion between the bodies themselves. 0 = no bath')
    ap.add_argument('--bath-sigma', dest='bath_sigma', type=float, default=3.0, help='that bath\'s hard-sphere diameter (nm)')
    ap.add_argument('--bath-d', dest='bath_d', type=float, default=0.15, help="that bath's diffusion coefficient (um^2/s); the collective "
                    'CFL 6q(1+8phi) < 1 binds it')
    ap.add_argument('--d-crowd', dest='d_crowd', type=float, default=None, help='um^2/s; default 0.04 (smear, ribosome-like) / 0.3 (point)')
    ap.add_argument('--free-volume', dest='free_volume', action='store_true', help='smear: arm the free-volume functional for the bath statics around the bodies')
    ap.add_argument('--phi-core', dest='phi_core', type=float, default=0.15)
    ap.add_argument('--phi-edge', dest='phi_edge', type=float, default=0.02)
    ap.add_argument('--r-edge-nm', dest='r_edge_nm', type=float, default=250.0, help='radius where the crowder fraction reaches phi_edge; flat beyond')
    ap.add_argument('--well-k', dest='well_k', type=float, default=None, help='kT/nm^2: replace the capped well by the uncapped harmonic psi = k r^2 / 2 (the shape an off-lattice sphere potential can build)')
    ap.add_argument('--n-crowders', dest='n_crowders', type=int, default=None, help='fix the crowder count (default: from the profile integral)')
    ap.add_argument('--start', default='ideal', choices=('ideal', 'uniform', 'lda'), help='initial crowder placement: on the ideal-gas profile e^-psi (default), uniform, or on the Carnahan-Starling local-density profile (a start from the wrong side / near equilibrium: tests whether the core relaxes)')
    ap.add_argument('--n-monomers', dest='n_monomers', type=int, default=60000)
    ap.add_argument('--kappa', type=float, default=4.0)
    ap.add_argument('--k-nuc', dest='k_nuc', type=float, default=0.02, help='order 1: 1/(s monomer) (a free monomer becomes a seed); order 2: 1/(s pair) at a voxel')
    ap.add_argument('--nuc-order', dest='nuc_order', type=int, default=1, choices=(1, 2),
                    help='1 (default at sigma = h): monomer seed, grows by tip addition; 2: dimer from two monomers in ONE voxel, which two 10 nm spheres almost never share (the pair state is excluded ~50x)')
    ap.add_argument('--k-poly', dest='k_poly', type=float, default=20.0, help='1/(s monomer) per accessible neighbour, both tips; the bending weights '
                    'leave ~0.4 effective monomers per tip at 0.24 M/voxel, so 20 gives ~8/s per tip against depoly 3/s (a smoke run at 4 lost every nucleus)')
    ap.add_argument('--k-depoly', dest='k_depoly', type=float, default=3.0, help='1/s per tip')
    ap.add_argument('--k-sever', dest='k_sever', type=float, default=0.05, help='1/(s subunit): fragmentation anywhere')
    ap.add_argument('--rate-scale', dest='rate_scale', type=float, default=1.0,
                    help='multiply every chemical rate (nucleation, growth, tip loss, severing) by this. With no drive the '
                         'stationary state is set by the rate RATIOS, so scaling them all leaves it unchanged and reaches it '
                         'x faster: the master equation is linear in time. Keep k tau << 1 (at tau 20 us, scale 10 puts the '
                         'growth propensity at ~2e-3 per step).')
    ap.add_argument('--no-chemistry', dest='chemistry', action='store_false')
    ap.add_argument('--seed-chains', dest='seed_chains', type=int, default=0, help='pre-place this many exact FCC chains (control: with --no-chemistry)')
    ap.add_argument('--seed-n', dest='seed_n', type=int, default=11)
    ap.add_argument('--no-rigid-translation', dest='rigid_translation', action='store_false', help='debug: no collective hops (chains and bodies)')
    ap.add_argument('--no-rigid-rotation', dest='rigid_rotation', action='store_false', help='debug: no rigid rotations')
    ap.add_argument('--si-moves', dest='si_moves', action='store_true', help='the SI move set: full kink, end-bond rotation, edge '
                    'translation of a segment (interior mask 0x1B) and the collective FACE translation; no rigid rotation')
    ap.add_argument('--interior-sweep', dest='interior_sweep', action='store_true', help='per-subunit interior sweep (one attempt per unit per step)')
    ap.add_argument('--chain-hop-scale', dest='chain_hop_scale', type=float, default=None,
                    help='multiply the CHAIN species collective hop only (0 = off; the crowder bodies keep theirs)')
    ap.add_argument('--subhop-q', dest='subhop_q', type=int, default=None, help='debug: compact-body sub-voxel hop Q (0 = whole-voxel path); default = engine default')
    # CEILING PROBE (opt-in; default off leaves every existing run byte-identical). Arms the
    # adaptive-tau controller, which holds the lattice-wide maximum of the per-pass hop propensity
    # at eps and so measures the operative timestep ceiling. --hop-sensor 0 restricts that maximum
    # to the subvolume hop, which is what the engine did before the body hop reported into it; the
    # pair of runs is the measurement of how much the oversize body tightens the ceiling.
    # Select the hop branch WITHOUT arming the controller, so the step stays at --tau-us and the
    # only thing that changes is how the two directional probabilities are formed. The engine
    # default is LINEAR, which is valid only while P+ + P- <= 1; EXACT is the bounded Poisson
    # tau-leap 1-exp(-W tau), which cannot overflow at any energy. This is the A/B that says
    # whether a scene sitting past the first-order bound actually gets a different answer.
    ap.add_argument('--track-bodies', dest='track_bodies', action='store_true',
                    help='record every crowder body centre of mass per frame (RAW, wrapped voxels), for the '
                         'mean-square displacement. Off by default: it is only needed by the transport arms.')
    ap.add_argument('--exact-hop', dest='exact_hop', action='store_true',
                    help='use the bounded exact hop branch instead of the first-order one')
    ap.add_argument('--ceiling-eps', dest='ceiling_eps', type=float, default=None,
                    help='arm adaptive tau at this budget and record the ceiling (default: off)')
    ap.add_argument('--hop-sensor', dest='hop_sensor', type=int, default=1,
                    help='1 = the body hop reports into the ceiling sensor (default); 0 = subvolume hop only')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='gradient_growth.npz')
    GB.add_args(ap)
    return ap.parse_args()


def radial_well(nx, phi_core, phi_edge, r_edge_nm, well_k=None):
    """psi[v] (kT) at voxel centres, index v = z*N*N + y*N + x (the engine's latticeIndex); r from the box centre.
    Default: A r^2 / r_e^2 capped at r_e (flat beyond), A = ln(phi_core / phi_edge). well_k: the uncapped harmonic
    k r^2 / 2 (kT/nm^2), the shape an off-lattice harmonic sphere potential reproduces exactly (ReaDDy reference)."""
    idx = np.arange(nx)
    z, y, x = np.meshgrid(idx, idx, idx, indexing='ij')
    c = (nx - 1) / 2.0
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) * H_NM
    if well_k is not None:
        A = float(0.5 * well_k * r_edge_nm ** 2)                       # the depth at r_e, for the log line
        psi = 0.5 * well_k * r ** 2
    else:
        A = float(np.log(phi_core / phi_edge))
        psi = A * np.minimum(r, r_edge_nm) ** 2 / r_edge_nm ** 2
    return psi.reshape(-1).astype(np.float32), r.reshape(-1), A


def shell_index(nx):
    """10 nm radial shells from the box centre (voxel resolution), for the radial histograms."""
    _, r, _ = radial_well(nx, 0.15, 0.02, 1.0)
    return np.minimum((r / H_NM).astype(np.int64), nx // 2 - 1)


def build_sim(args, psi, r_vox):
    nx = args.nx
    box_um = nx * H_NM * 1e-3
    sim = RDME.RDMESimulation(dimensions=micron(box_um, box_um, box_um), spacing=nm(H_NM))
    smear = args.crowder_mode == 'smear'
    species = ['M', 'R' if smear else 'C']
    if smear and args.bath_eta > 0: species.append('B')             # the ternary scene: 20 nm bodies AND a small bath
    sim.defineSpecies(species)
    sim.modifyRegion('default').setDefaultDiffusionRate(args.d_mono * 1e-12)
    sim.modifyRegion('default').setDiffusionRate('R' if smear else 'C', args.d_crowd * 1e-12)   # R: no byte particles; the row must exist
    if smear and args.bath_eta > 0:
        sim.modifyRegion('default').setDiffusionRate('B', args.bath_d * 1e-12)
    v_c = (np.pi / 6.0) * args.sigma_crowd ** 3
    phi_v = args.phi_core * np.exp(-psi.astype(np.float64))                 # target profile (ideal-gas response to the well)
    n_c = int(round(np.sum(phi_v) * H_NM ** 3 / v_c)) if args.n_crowders is None else int(args.n_crowders)
    n_m = args.n_monomers if args.chemistry else 0
    if n_m > 0: sim.addParticles('M', 'default', n_m)                      # monomers dispersed uniformly at t = 0 (the spec)
    if smear and args.bath_eta > 0:
        n_b = int(round(args.bath_eta * (nx * H_NM) ** 3 / ((np.pi / 6.0) * args.bath_sigma ** 3)))
        sim.addParticles('B', 'default', n_b)
        print(f"[gradient] ternary: {n_b} bath particles of {args.bath_sigma} nm at eta {args.bath_eta} "
              f"(D {args.bath_d} um^2/s, collective CFL 6q(1+8phi)={6 * args.bath_d * 1e6 * args.tau_us * 1e-6 / H_NM ** 2 * (1 + 8 * args.bath_eta):.2f})", flush=True)
    if n_c > 0 and not smear:
        # Crowders placed ON the well's equilibrium profile at t = 0 (multinomial over voxels with weight exp(-psi)), so the
        # gradient is present from the first step instead of forming over ~L^2/D. addParticleAtIdx needs the discretised lattice.
        sim.getLattice()
        pv = phi_v / phi_v.sum()
        rng = np.random.default_rng(args.seed + 7919)
        vox = rng.choice(len(pv), size=n_c, p=pv)
        for v in vox:
            z, rem = divmod(int(v), nx * nx); y, x = divmod(rem, nx)
            sim.addParticleAtIdx((x, y, z), 'C')
    # The engine reads write / lattice-write / hook intervals as STEP COUNTS: frames come from the hook, writes never.
    tau_s = args.tau_us * 1e-6
    hook_steps = max(1, int(round(args.frame_dt / tau_s)))
    scene.finish_sim(sim, args.seed, tau_s, args.t_end, hook_steps, scene.NEVER_STEPS, scene.NEVER_STEPS)
    return sim, n_m, n_c, phi_v


from gradient_hook import Recorder                               # noqa: E402  (the recording hook: one frame of observables)

def place_bodies(n, phi_v, nx, rng, min_sep=2, blocked=None):
    """n body centres drawn on the well's profile (weight phi_v per voxel), no two within min_sep voxels on any axis:
    20 nm spheres at contact are 2 voxels apart, so min_sep 2 is the no-overlap condition at t = 0."""
    pv = phi_v / phi_v.sum()
    taken = np.zeros((nx, nx, nx), bool) if blocked is None else blocked.copy()   # `blocked`: voxels already spoken for
    pts = []
    guard = 0
    while len(pts) < n and guard < 200 * n + 100000:
        guard += 1
        v = int(rng.choice(len(pv), p=pv))
        z, rem = divmod(v, nx * nx); y, x = divmod(rem, nx)
        lo = [max(0, c - (min_sep - 1)) for c in (x, y, z)]; hi = [min(nx, c + min_sep) for c in (x, y, z)]
        if taken[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any(): continue
        taken[x, y, z] = True; pts.append((x, y, z))
    if len(pts) < n: raise RuntimeError(f"placed {len(pts)}/{n} bodies: the core is too dense for 20 nm spheres at contact")
    return pts


def configure(sol, args, psi, rng, phi_v):
    nx = args.nx
    xyz = nx ** 3
    smear = args.crowder_mode == 'smear'
    crowd_id = R_ID if smear else C_ID
    sol.setPotentialBasis(psi, np.zeros(xyz, np.float32))                  # basis 0 = the crowder well, basis 1 unused
    sol.setSpeciesCoupling(crowd_id - 1, 1.0, 0.0)                         # crowders (byte particles or bodies) feel the well
    sol.setSpeciesCoupling(M_ID - 1, 0.0, 0.0)                             # monomers do not (they partition by exclusion)
    sol.setStructureHandler(1, args.sigma, H_NM ** 3)                       # c = floor(h / sigma) = 1: one subunit per voxel
    sol.configureStructureLattice(nx, nx, nx)
    diam = [args.sigma, SIGMA_BODY_ROW if smear else args.sigma_crowd]
    if smear and args.bath_eta > 0: diam.append(args.bath_sigma)     # the ternary bath is a third byte species
    sol.setSpeciesHardSphereDiameters(np.array(diam, np.float64), H_NM ** 3)
    sol.setHardCoreExclusion(0.0 if args.no_vex else 1.0, WB)
    if not args.no_vex:
        sol.setStructureReciprocalVex(True, M_ID)                           # the subunit IS an M in the field
    else:
        sol.setStructureReciprocalVex(False, M_ID)
        sol.setStructureBodyPairExclusion(0)
    if os.environ.get('GRAD_NO_BEAD_CORE'):                                 # ablation: the pre-2026-09-13 coupling, subunits feeling the bodies' smeared field
        sol.setStructureBeadBodyExclusion(0)
    if not args.chemistry or args.n_monomers == 0:
        # No monomers on the lattice: the chain friction OFF. It rate-couples every byte type with sigma > 0 and D > 0 and
        # scans a 26-voxel shell per moved subunit per attempt to read its occupancy; with nothing to read that scan was
        # 0.5-0.8 ms per step of pure latency (8 chains, 2026-09-11 profile). Off, the empty lattice enters the move energy
        # instead, where it contributes exactly zero; a static M row is refused by the auto-coefficient check. Repeated in
        # the first hook: runSolver's init drops the setting.
        sol.setStructureChainFriction(0)
    n_bodies = 0
    if smear and args.n_crowders == 0:
        smear = False                                                        # an explicit zero: the dry reference of a crowder ladder, no bodies and no field
        print('[gradient] no crowder bodies requested: dry reference (no smear field)', flush=True)
    if smear:
        v_c = (np.pi / 6.0) * args.sigma_crowd ** 3
        n_bodies = int(round(np.sum(phi_v) * H_NM ** 3 / v_c)) if args.n_crowders is None else int(args.n_crowders)
        w_place = phi_v
        if args.start == 'uniform':
            w_place = np.ones_like(phi_v)
        elif args.start == 'lda':                                            # the CS profile at this N (bd_radial_well.cs_lda_profile)
            import bd_radial_well as BW
            n_sh = nx // 2; sv = BW.shell_voxels(nx, n_sh).astype(float)
            shells_v = shell_index(nx)                                       # psi per shell = the voxel field averaged over the shell
            psi_sh = np.bincount(shells_v, weights=psi.astype(np.float64), minlength=n_sh)[:n_sh] / np.maximum(np.bincount(shells_v, minlength=n_sh)[:n_sh], 1)
            lda = BW.cs_lda_profile(psi_sh, sv * H_NM ** 3, v_c, n_bodies)
            w_place = lda[shells_v]
        # The chains go down FIRST and the bodies are placed around them: a body drawn into a chain's voxel, or the ring
        # next to it, would start the run overlapping a bead (20 nm centre within one voxel of a 10 nm bead), which no
        # move can undo. FX.seed_many is called once here and the same chains are written into the overlay below.
        seeded = GB.seeded_chains(args, None) or (list(FX.seed_many(args.seed_chains, args.seed_n, args.kappa, nx, nx, nx, rng)) if args.seed_chains > 0 else [])
        blocked = np.zeros((nx, nx, nx), bool)
        for pos, _ in seeded:
            for q in pos:
                x0, y0, z0 = int(q[0]), int(q[1]), int(q[2])
                blocked[max(0, x0 - 1):x0 + 2, max(0, y0 - 1):y0 + 2, max(0, z0 - 1):z0 + 2] = True
        pts = GB.bodies(args, n_bodies, lambda: place_bodies(n_bodies, w_place, nx, rng, blocked=blocked))
        l1 = np.zeros(xyz, np.int32); l2 = np.full(xyz, DIR_NONE, np.int32); l3 = np.zeros(xyz, np.int32)
        coms, flags, nsubs = [], [], []
        for i, (x, y, z) in enumerate(pts):
            v = z * nx * nx + y * nx + x
            l1[v] = i + 1; l3[v] = (ROLE_BODY << 4) | 1
            coms += [float(x), float(y), float(z)]; nsubs.append(1); flags.append(FLAG_ACTIVE | FLAG_RIGID | FLAG_INERT | (R_ID << 8))   # a crowder body: rigid (no interior kernel), inert (no chemistry)
        # Seeded chains (if any) share this overlay and registry: H.seed_chains writes a FRESH overlay and registry, which
        # would wipe the bodies -- the run would then start with no crowders at all (and the body-bath friction would have
        # nothing to derive its coefficients from, which is how this was caught).
        chain_vox = []
        if seeded:
            for pos, dirs in seeded:
                vox = [int(q[0]) + nx * int(q[1]) + nx * nx * int(q[2]) for q in pos]
                if any(l1[v] for v in vox):
                    raise RuntimeError('a seeded chain overlaps a crowder body: lower --seed-chains or the crowder fraction')
                cid = n_bodies + len(chain_vox) + 1
                for i, v in enumerate(vox):
                    l1[v] = cid; l3[v] = (ROLE_BODY << 4) | 1
                    if i < len(vox) - 1: l2[v] = int(dirs[i])
                coms += np.asarray(pos, np.float64).mean(axis=0).tolist()
                nsubs.append(len(vox)); flags.append(FLAG_ACTIVE | (M_ID << 8))     # flexible: no RIGID, no INERT
                chain_vox.append(vox)
        sol.setStructureOverlay(l1, l2, l3)
        sol.setStructureRegistry(n_bodies + len(chain_vox), np.array(nsubs, np.int32), np.array(coms, np.float64), np.array(flags, np.int32))
        for i, (x, y, z) in enumerate(pts):
            v = z * nx * nx + y * nx + x
            sol.setStructureTips(i, int(v), int(v), -1)
        for k, vox in enumerate(chain_vox):
            sol.setStructureTips(n_bodies + k, vox[0], vox[-1], -1)
        if not os.environ.get('GRAD_NO_SMEAR'):                             # debug switch: bodies registered but no footprint
            sol.setStructureSmearStencil(rb.fmt_stencil(args.sigma_crowd, H_NM).reshape(-1).astype(np.float32))
            sol.setStructureSmearDnaBase(np.zeros(4 * xyz, np.float64))      # empty baseline: the bodies deposit live
            sol.armStructureSmearRefresh(R_ID)
            if not args.no_vex:
                sol.setStructureSmearExtAll(1)                               # every chain subunit feels the bodies' footprint (energy)
            if args.free_volume: sol.setStructureFreeVolume(1)
    cfg = H.ARMS['edge']
    sol.setStructureStretch(cfg['k_stretch'], cfg['rest_len'])              # FCC: the edge orbit
    if os.environ.get('GRAD_DU_HIST'): sol.setStructureHopEnergyHistogram(1)
    if os.environ.get('GRAD_ESCAPE_RATE'): sol.setStructureHopEscapeRate(True)   # debug: the uncapped WPE escape-rate hop (whole-voxel path)
    tau_s = args.tau_us * 1e-6
    att = max(1, int(round(10 * tau_s / 50e-6)))                            # the 50 us convention: 10 attempts per complex per 50 us
    sol.setStructureLaunchParams(att, float(args.kappa))
    d1_nm2 = args.d_mono * 1e6
    if args.rigid_translation:
        sol.setStructureSizeDiffusion(0, d1_nm2, tau_s, H_NM)               # Rouse D(N) = D1/N from the live n (bodies: n = 1 -> D1 ...)
        if smear: sol.setStructureHopSpeciesScale(R_ID, args.d_crowd / args.d_mono)   # ... times the body species' scale = D_body
    if args.si_moves:
        args.rigid_rotation = False                                          # not in the SI set
        sol.setStructureInteriorMoveMask(0x1B)                               # end rotation | flip | sideways | edge translation
    if args.interior_sweep: sol.setStructureInteriorSweep(1)
    if args.chain_hop_scale is not None: sol.setStructureHopSpeciesScale(M_ID, args.chain_hop_scale)
    if args.rigid_rotation:
        sol.setStructureRotationModel(1, d1_nm2, tau_s, H_NM)
    if args.chemistry:
        sol.addStructureChannel(RXN_NUCLEATE, M_ID, 0, int(args.nuc_order), args.k_nuc, 0.0, 0)
        sol.addStructureChannel(RXN_POLYMERIZE, M_ID, 0, 1, args.k_poly, 0.0, 0)       # both tips (unified tip kernel)
        sol.addStructureChannel(RXN_DEPOLYMERIZE, 0, M_ID, 1, args.k_depoly, 0.0, 0)   # - tip retract + reversible + tip shed
        if args.k_sever > 0: sol.addStructureChannel(RXN_SEVER, M_ID, M_ID, 1, args.k_sever, 0.0, 0)
    if (args.seed_chains > 0 or args.state) and n_bodies == 0:                 # with bodies the chains were seeded above
        chains = GB.seeded_chains(args, None) or FX.seed_many(args.seed_chains, args.seed_n, args.kappa, nx, nx, nx, rng)
        H.seed_chains(sol, chains, nx, nx, nx, species=M_ID)
    return att, n_bodies


def main():
    args = parse_args()
    if args.sigma > H_NM: sys.exit("--sigma must not exceed the voxel edge")
    smear = args.crowder_mode == 'smear'
    if args.sigma_crowd is None: args.sigma_crowd = 20.0 if smear else 10.0
    if args.d_crowd is None: args.d_crowd = 0.04 if smear else 0.3
    if smear and args.sigma_crowd <= H_NM: sys.exit("smear mode wants a crowder larger than the voxel (the footprint must reach the neighbours)")
    nx = args.nx
    psi, r_nm, A = radial_well(nx, args.phi_core, args.phi_edge, args.r_edge_nm, args.well_k)
    q_c = args.d_crowd * 1e6 * args.tau_us * 1e-6 / H_NM ** 2
    cfl = 6.0 * q_c * (1.0 + 8.0 * args.phi_core)
    print(f"[gradient] well A={A:.3f} kT, phi {args.phi_core}->{args.phi_edge} at r={args.r_edge_nm} nm; crowder q={q_c:.3f}/dir, "
          f"6q(1+8phi_core)={cfl:.2f} (must be < 1: the bath transport runs away above)", flush=True)
    if cfl >= 1.0 and not smear: sys.exit("crowder transport unstable at this density and step: lower --d-crowd, --phi-core or --tau-us")
    if args.sigma_crowd < args.sigma:
        print(f"[gradient] WARNING: crowders ({args.sigma_crowd} nm) smaller than the monomer ({args.sigma} nm): the free monomer is oversize "
              f"in the byte-lattice hop and self-traps in dense voxels until the rate factor is extended to byte-lattice species", flush=True)
    if args.rate_scale != 1.0:
        for k in ('k_nuc', 'k_poly', 'k_depoly', 'k_sever'):
            setattr(args, k, getattr(args, k) * args.rate_scale)
        print(f"[gradient] rate scale {args.rate_scale}: nuc {args.k_nuc:g} poly {args.k_poly:g} depoly {args.k_depoly:g} "
              f"sever {args.k_sever:g} (the stationary state is unchanged; it arrives {args.rate_scale}x sooner)", flush=True)
    rng = np.random.default_rng(args.seed)
    GB.apply(args)                                                   # bench clock + --state-from, before the scene is built
    sim, n_m, n_c, phi_v = build_sim(args, psi, r_nm / H_NM)
    GB.lattice_frames(sim, args)
    # The chain's rate-coupled bath: the byte-lattice crowders (point mode) or the monomers themselves (smear mode, where the
    # bodies act on the chain as an energy). Report the closure for that bath.
    bath_sigma, bath_d = (args.sigma, args.d_mono) if smear else (args.sigma_crowd, args.d_crowd)
    c_closure = FC.coefficient(args.sigma, bath_sigma, args.d_mono, bath_d)
    phi_m = (n_m / nx ** 3) * (np.pi / 6.0) * args.sigma ** 3 / H_NM ** 3
    print(f"[gradient] box {nx}^3, tau {args.tau_us} us, monomers {n_m} (phi_M {phi_m:.3f}), crowders {n_c} (expected phi_core {args.phi_core}); "
          f"chain friction closure c={c_closure:.2f} against {'the monomers' if smear else 'the crowders'} => "
          + (f"m(monomer bath)={FC.mobility(phi_m, c_closure):.3f}" if smear else f"m(core)={FC.mobility(args.phi_core, c_closure):.3f}, m(edge)={FC.mobility(args.phi_edge, c_closure):.3f}")
          + f"; chemistry={'on' if args.chemistry else 'OFF'} seed chains={args.seed_chains}", flush=True)
    path = args.out.replace('.npz', '.lm')
    if os.path.exists(path): os.remove(path)
    if not GB.active(args): sim.save(path)
    lm.setVerbosityLevel(int(os.environ.get('LM_VERBOSITY', '1')))
    sol = Recorder()
    att, n_bodies = configure(sol, args, psi, rng, phi_v)
    if args.exact_hop:
        sol.setHopIntegration(1)
        print('[gradient] EXACT hop branch armed (bounded 1-exp(-W tau)); step unchanged', flush=True)
    if args.ceiling_eps is not None:
        sol.setHopIntegration(1)
        try:
            sol.setStructureHopSensor(int(args.hop_sensor))
        except AttributeError:
            print('[ceiling] this build has no setStructureHopSensor: the body hop cannot report, '
                  'so the ceiling below is the subvolume hop alone', flush=True)
        sol.setAdaptiveTau(1, float(args.ceiling_eps), 0.05)
    shells = shell_index(nx); n_shells = nx // 2
    sol.configure_recording(args, shells, n_shells, n_bodies)
    sol.save_beads = bool(args.save_beads)
    if args.snap_frames > 0:
        n_fr = int(round(args.t_end / args.frame_dt))
        sol.snap_at = set(np.unique(np.linspace(0, n_fr, args.snap_frames).astype(int)).tolist())
    print(f"[gradient] mode={args.crowder_mode} bodies={n_bodies} (sigma {args.sigma_crowd} nm, D {args.d_crowd} um^2/s, free_volume={args.free_volume}); "
          f"attempts/step={att} kappa={args.kappa} channels: nuc {args.k_nuc} poly {args.k_poly} depoly {args.k_depoly} sever {args.k_sever}", flush=True)
    gpu = int(os.environ.get('STRUCT_GPU', '0'))
    t0 = time.perf_counter()
    if GB.active(args): wall = GB.run(sim, sol, gpu, args)
    else:
        lm.runSolver(path, 1, sol, [gpu], 0)
        wall = time.perf_counter() - t0
    err = np.zeros(4, np.int32); sol.getStructureError(err)
    ceiling = {}
    if args.ceiling_eps is not None:
        ts = float(sol.getTauScale()); sm = float(sol.getLastSumMax())
        ceiling = dict(eps=float(args.ceiling_eps), hop_sensor=int(args.hop_sensor),
                       tau_scale=ts, sum_max=sm, tau_nominal_s=args.tau_us * 1e-6,
                       tau_ceiling_s=args.tau_us * 1e-6 * ts)
        print('[ceiling] eps=%.3g hop_sensor=%d -> tau_scale %.4g, sum_max %.4g, ceiling %.4g us'
              % (args.ceiling_eps, args.hop_sensor, ts, sm, ceiling['tau_ceiling_s'] * 1e6), flush=True)
    if os.environ.get('STRUCT_PROFILE'): sol.printStructureProfile()
    if os.environ.get('GRAD_DU_HIST'):                                       # debug: the hop kernel's move-energy histogram
        h = np.zeros(2 * 512, np.float64); sol.getStructureHopEnergyHistogram(h)
        edges = -20.0 + 0.5 * (np.arange(512) + 0.5)
        for name, hh in (('attempted', h[:512]), ('accepted', h[512:])):
            n = hh.sum(); m = (hh * edges).sum() / max(n, 1); neg = hh[edges < -0.25].sum() / max(n, 1); pos = hh[edges > 0.25].sum() / max(n, 1)
            print(f"[du-hist] {name}: n={int(n)} mean dU={m:.3f} kT  frac dU<-0.25: {neg:.3f}  frac dU>+0.25: {pos:.3f}  underflow {int(hh[0])} overflow {int(hh[-1])}", flush=True)
    fric = np.zeros(3, np.float64)
    try: sol.getStructureChainFrictionStat(fric)
    except Exception: fric[:] = np.nan
    if os.environ.get('GRAD_TRACE_BODIES'):                                  # debug: did the bodies take the sub-voxel path at all?
        bf = np.zeros(3, np.float64)
        try: sol.getStructureFrictionStat(bf); print(f"[bodies] body friction stat (sub-voxel path decisions, mean m, shell voxels): {bf.tolist()}", flush=True)
        except Exception as e: print(f"[bodies] body friction stat unavailable: {e}", flush=True)
    bb = sol.last_backbones or {}
    bb_cid = np.array(sorted(bb.keys()), np.int64)
    bb_len = np.array([len(bb[c]) for c in bb_cid], np.int64)
    bb_pos = np.concatenate([bb[c] for c in bb_cid], axis=0) if len(bb_cid) else np.zeros((0, 3), np.int64)
    import json as _json
    np.savez_compressed(args.out, ceiling=_json.dumps(ceiling), times=np.array(sol.times), n_chains=np.array(sol.n_chains), subunits=np.array(sol.subunits),
                        free_m=np.array(sol.free_m), n_crowd=np.array(sol.n_crowd), len_hist=np.array(sol.len_hist),
                        rad_m=np.array(sol.rad_m), rad_c=np.array(sol.rad_c), rad_com=np.array(sol.rad_com), rad_sub=np.array(sol.rad_sub),
                        nucleated=np.array(sol.nucleated), shell_nm=(np.arange(n_shells) + 0.5) * H_NM, shell_voxels=np.bincount(shells, minlength=n_shells)[:n_shells],
                        psi=psi, phi_target=phi_v.astype(np.float32), bb_cid=bb_cid, bb_len=bb_len, bb_pos=bb_pos,
                        com_rows=np.array(sol.com_rows, np.float64).reshape(-1, 6),   # (frame, cid, n, x, y, z): the chain trajectories
                        **({'body_com': np.asarray(sol.body_com, np.float32)} if sol.body_com else {}),   # (frame, body, 3) WRAPPED voxels
                        bound_fill=np.array(sol.bound_fill), chain_voxels=np.array(sol.chain_voxels),
                        r2_ee=np.array(sol.r2_ee), r2_g=np.array(sol.r2_g), n_sized=np.array(sol.n_sized),
                        rad_len_sum=np.array(sol.rad_len_sum),
                        demix_r=np.array(sol.demix_r), demix_contrast=np.array(sol.demix_contrast),
                        demix_crowd_var=np.array(sol.demix_crowd_var),
                        **({'beads_vox': np.stack(sol.beads)} if sol.beads else {}),
                        **{k: v for i, (ts, bv, cv, ba) in enumerate(sol.snaps)
                           for k, v in ([('snap_t_%d' % i, ts), ('snap_body_%d' % i, bv), ('snap_chain_%d' % i, cv)]
                                        + ([('snap_bath_%d' % i, ba)] if ba is not None else []))},
                        demix_cells=np.array(sol.demix_cells),
                        bead_body_hist=sol.bead_body_hist,
                        overlay_l1=sol.last_overlay[0], overlay_l2=sol.last_overlay[1], overlay_l3=sol.last_overlay[2],
                        friction_decisions=fric[0], friction_mean_m=fric[1], friction_shell=fric[2], c_closure=c_closure,
                        n_monomers=n_m, n_crowders=(n_bodies if smear else n_c), n_bodies=n_bodies, struct_err=int(err[0]), wall_s=wall, frame_hash=np.array(sol.frame_hash), attempts=att,
                        **{'arg_' + k: v for k, v in vars(args).items()})       # every parameter, prefixed to avoid collisions
    last = sol.subunits[-1] if sol.subunits else 0
    bf = sol.bound_fill[-1] if sol.bound_fill else 0
    print(f"[gradient] done: {len(sol.times)} frames, wall {wall:.0f}s, struct_err={int(err[0])}, chains={sol.n_chains[-1] if sol.n_chains else 0}, "
          f"bound={last} (overlay fill {bf} in {sol.chain_voxels[-1] if sol.chain_voxels else 0} voxels), free M={sol.free_m[-1] if sol.free_m else 0} "
          f"(mass {bf + (sol.free_m[-1] if sol.free_m else 0)} / {n_m}), "
          f"mean m={fric[1]:.3f} over {int(fric[0]) if np.isfinite(fric[0]) else 0} decisions -> {args.out}")
    if int(err[0]) != 0: sys.exit(f"struct_err={int(err[0])}")


if __name__ == '__main__':
    main()
