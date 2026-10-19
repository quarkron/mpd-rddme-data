#!/usr/bin/env python3
"""Off-lattice reference for Fig 4: the semiflexible chain in an explicit bath of spheres (ReaDDy, free-draining).

--bath-sigma sets the depletant: 0.3 voxels is the 3 nm bath this was written for, 2.0 voxels is a 20 nm crowder body,
the off-lattice check of the exact bead-body hard core (pass --d-ratio 0.1333 to give it its own 0.04 um^2/s).

The chain is readdy_chain.py's (bonds, angles, bead repulsion, matched to the lattice on bond length, persistence length,
subunit D and bead diameter); the bath is readdy_msd.py's (harmonic excluded volume at contact, periodic box). Units are
lattice voxels (10 nm) and seconds, so every number compares to a lattice number without conversion.

WHAT IS MATCHED. Bond length sqrt 2 vox, persistence length L_p/b = fcc_exact.lp_over_b(kappa) (4.4005 at kappa 4), bead D_1
= 0.3 um^2/s = 3000 vox^2/s, bead diameter 1 vox; bath diameter 0.3 vox, D_B = D_1 by default (--d-ratio; the closure carries
D_1/(D_1 + D_B), so the lattice and this reference must share the ratio), packing fraction --eta. The angle force constant
comes from a root find on the dry chain (--match, or pass --k-angle); nothing else is free. Bath-bead and bath-bath
repulsions use 50 kT/nm^2 = 5000 kT/vox^2 as in the validated tracer ladder (penetration ~0.14 nm at kT).

WHAT IS REDUCED. The box: the lattice ran 32^3 voxels (320 nm) to hold eight chains; one chain needs only a box larger than
its extent (R_ee ~ 10 vox at kappa 4, M = 10), so --box 16 (160 nm, ~43 000 bath spheres at eta 0.15) keeps the same
parameter set at a fraction of the cost. Time: --t-end 0.3 s covers >= 25 Rouse times at eta 0.15 (tau_1 ~ 11 ms there).

  python3 readdy_chain_bath.py --match --out kangle.json                 # once: k_angle for kappa 4 (dry, parallel)
  python3 readdy_chain_bath.py --eta 0.15 --k-angle K --probe 2000        # cost estimate
  python3 readdy_chain_bath.py --eta 0.15 --k-angle K --seed 1 --out RB_e015_s1.npz
Output npz: frames (F, n, 3) bead positions in voxels, dt_frame, box, eta, d_ratio, k_angle, wall time.
★THE FRAMES ARE NOT FULLY UNWRAPPED. They span several box lengths, so they are not plainly wrapped either, but about
4 % of frame-to-frame bead displacements are a box length (14.8 voxels against a median step of 0.33), which is the
periodic image showing through. Any analysis must unwrap frame to frame -- d -= box*round(d/box) -- before computing a
centre of mass or a displacement; without it D_cm comes out nine times too large and is not even monotone in eta.
"""
import argparse, json, os, sys, time
import numpy as np

os.environ.setdefault('CONDA_PREFIX', '/raid/racda/opt/readdy_env')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'Lattice_Microbes_2.6', 'testing', 'engine'))
import readdy_chain as RC                        # noqa: E402
import fcc_exact as FX                           # noqa: E402

SIG_B = 0.3                                      # default depletant: the 3 nm bath sphere, in voxels (--bath-sigma)
K_REP_BATH = 5000.0                              # 50 kT/nm^2 in voxel units (readdy_msd's force constant)


def bath_count(eta, box, sig=None):
    return int(round(eta * box ** 3 / ((np.pi / 6.0) * (SIG_B if sig is None else sig) ** 3)))


def build(n, k_angle, box, eta, d1, d_b, bond_len=RC.BOND_EDGE, k_bond=200.0, k_rep=200.0, sigma=RC.SIGMA):
    import readdy
    system = readdy.ReactionDiffusionSystem(box_size=[box, box, box], periodic_boundary_conditions=[True] * 3, unit_system=None)
    system.add_topology_species('M', d1)
    system.topologies.add_type('chain')
    system.topologies.configure_harmonic_bond('M', 'M', force_constant=k_bond, length=bond_len)
    system.topologies.configure_harmonic_angle('M', 'M', 'M', force_constant=k_angle, equilibrium_angle=np.pi)
    system.potentials.add_harmonic_repulsion('M', 'M', force_constant=k_rep, interaction_distance=sigma)
    if eta > 0.0:
        system.add_species('B', d_b)
        system.potentials.add_harmonic_repulsion('M', 'B', force_constant=K_REP_BATH, interaction_distance=(sigma + SIG_B) / 2.0)
        system.potentials.add_harmonic_repulsion('B', 'B', force_constant=K_REP_BATH, interaction_distance=SIG_B)
    return system


def main():
    global SIG_B                                          # --bath-sigma: every repulsion radius reads it
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=11)
    ap.add_argument('--kappa', type=float, default=4.0, help='lattice stiffness whose exact L_p/b is the match target')
    ap.add_argument('--k-angle', dest='k_angle', type=float, default=None)
    ap.add_argument('--match', action='store_true', help='root-find k_angle for --kappa on the dry chain and write it to --out (json)')
    ap.add_argument('--eta', type=float, default=0.15)
    ap.add_argument('--box', type=float, default=16.0, help='voxels')
    ap.add_argument('--d1', type=float, default=RC.D1_VOXEL2_PER_S, help='vox^2/s')
    ap.add_argument('--d-ratio', dest='d_ratio', type=float, default=1.0, help='D_B / D_1')
    ap.add_argument('--bath-sigma', dest='bath_sigma', type=float, default=SIG_B,
                    help='depletant diameter in voxels: 0.3 = the 3 nm bath (default), 2.0 = a 20 nm crowder body. With '
                         '2.0 pass --d-ratio 0.1333 so the crowder carries its own 0.04 um^2/s, and a box that holds the '
                         'chain: at eta 0.15 a 160 nm box needs 147 crowders against 43 000 bath spheres, so this is the '
                         'cheaper scene, not the dearer one.')
    ap.add_argument('--t-end', dest='t_end', type=float, default=0.3, help='seconds')
    ap.add_argument('--step-nm', dest='step_nm', type=float, default=0.3, help='rms 3-D step of the fastest species per dt')
    ap.add_argument('--frames', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--probe', type=int, default=0, help='time this many steps, extrapolate, exit')
    ap.add_argument('--out', default='readdy_chain_bath.npz')
    a = ap.parse_args()
    SIG_B = a.bath_sigma

    target = FX.lp_over_b(a.kappa)
    if a.match:
        t0 = time.perf_counter()
        res = RC.match_persistence_length_parallel(target)        # a dict: ok, k_angle, lp_over_b, err, ...
        if not res.get('ok'): sys.exit(f"[readdy] match failed: {res}")
        rec = dict(kappa=a.kappa, lp_over_b_target=float(target), k_angle=float(res['k_angle']), lp_over_b_got=float(res['lp_over_b']),
                   err=float(res['err']), wall_s=time.perf_counter() - t0)
        json.dump(rec, open(a.out, 'w'), indent=1); print(f"[readdy] matched k_angle = {rec['k_angle']:.4f} -> L_p/b {rec['lp_over_b_got']:.4f} +- {rec['err']:.4f} (target {target:.4f}, kappa {a.kappa}) in {rec['wall_s']:.0f} s -> {a.out}")
        return
    if a.k_angle is None:
        sys.exit("pass --k-angle (from --match) or --match")

    d_b = a.d1 * a.d_ratio
    d_fast = max(a.d1, d_b)
    step_vox = a.step_nm / RC.VOXEL_NM
    dt = step_vox ** 2 / (6.0 * d_fast)
    nsteps = int(round(a.t_end / dt)); stride = max(1, nsteps // a.frames)
    nb = bath_count(a.eta, a.box) if a.eta > 0 else 0
    print(f"[readdy] n={a.n} kappa={a.kappa} (L_p/b {target:.3f}) k_angle={a.k_angle} eta={a.eta} box={a.box:.0f} vox bath={nb} "
          f"D1={a.d1:g} D_B={d_b:g} vox^2/s dt={dt:.3e} s steps={nsteps} stride={stride}", flush=True)

    import readdy
    system = build(a.n, a.k_angle, a.box, a.eta, a.d1, d_b)
    sim = system.simulation(kernel="CPU")
    sim.show_progress = False
    try:
        sim.kernel_configuration.n_threads = a.threads
    except Exception as e:
        print(f"[readdy] n_threads not settable: {e}")
    rng = np.random.default_rng(a.seed)
    init = RC.seed_positions(a.n, RC.BOND_EDGE, a.box, rng, 'coil')
    top = sim.add_topology('chain', ['M'] * a.n, init)
    for i in range(a.n - 1):
        top.graph.add_edge(i, i + 1)
    if nb > 0:
        pos = (rng.random((nb, 3)) - 0.5) * a.box
        # keep the bath off the chain's beads at t = 0 (a sphere inside a bead would be launched by the repulsion)
        d = np.min(np.linalg.norm(pos[:, None, :] - init[None, :, :], axis=2), axis=1)
        pos = pos[d > 0.5 * (RC.SIGMA + SIG_B)]
        while len(pos) < nb:
            extra = (rng.random((nb - len(pos), 3)) - 0.5) * a.box
            d = np.min(np.linalg.norm(extra[:, None, :] - init[None, :, :], axis=2), axis=1)
            pos = np.concatenate([pos, extra[d > 0.5 * (RC.SIGMA + SIG_B)]])
        sim.add_particles('B', pos[:nb])
    frames = []
    sim.observe.particle_positions(stride, types=['M'], callback=lambda x: frames.append(np.array([[p[0], p[1], p[2]] for p in x])), save=False)
    if a.probe:
        t0 = time.perf_counter(); sim.run(a.probe, dt, show_summary=False); el = time.perf_counter() - t0
        print(f"[readdy] PROBE {a.probe} steps in {el:.1f} s -> {el / a.probe * 1e3:.3f} ms/step; full run {nsteps} steps ~ {el / a.probe * nsteps / 3600:.2f} h on {a.threads} threads")
        return
    t0 = time.perf_counter(); sim.run(nsteps, dt, show_summary=False); wall = time.perf_counter() - t0
    fr = np.array([RC._unwrap(f, a.box) for f in frames])
    np.savez_compressed(a.out, frames=fr, dt_frame=dt * stride, dt=dt, box=a.box, eta=a.eta, n=a.n, kappa=a.kappa, k_angle=a.k_angle,
                        d1=a.d1, d_b=d_b, d_ratio=a.d_ratio, n_bath=nb, seed=a.seed, wall_s=wall, step_nm=a.step_nm)
    print(f"[readdy] done: {len(fr)} frames, wall {wall:.0f} s -> {a.out}")


if __name__ == '__main__':
    main()
