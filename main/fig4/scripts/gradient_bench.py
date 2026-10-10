"""Bench mode, lattice reuse and state seeding for gradient_growth.py, for the ERA structure loop's points.

The loop drives every point the same way (tools/points.sh run_point): exactly --bench steps with a frame every
--frame-steps, from a saved initial lattice (--reuse-lm / --save-init-lm), keeping the finished lattice for the identity
gate (--keep-lm, --lattice-frames), and hashing each frame. bath_scene.add_scene_args carries those options for the
engine drivers, but it also defines --seed, which gradient_growth.py already has; the subset lives here instead, so the
driver itself gains only hook lines.

--state-from <npz> starts the scene from a finished run of this driver: its final chains bead for bead (bb_len /
bb_pos, voxel coordinates in backbone order) and the bodies of its last snapshot (snap_body_<k>, linear voxel index, x
fastest), with the unbound monomers placed free as usual. That is how a bench point measures the DEVELOPED demixing
scene (about 2 650 chains holding 45 000 of the 60 000 subunits at 10 s) without waiting ten simulated seconds; the
exact chain sampler cannot seed it, since it caps a 64^3 box at 343 chains.
"""
import hashlib

import numpy as np

import bath_scene as scene


def add_args(ap):
    ap.add_argument('--bench', type=int, default=0, metavar='STEPS',
                    help='bench mode: run exactly STEPS timesteps with one frame every --frame-steps (overrides --t-end/--frame-dt)')
    ap.add_argument('--frame-steps', dest='frame_steps', type=int, default=0, help='bench mode: steps per frame')
    ap.add_argument('--reuse-lm', dest='reuse_lm', default=None, help='start from this prebuilt lattice file (must exist)')
    ap.add_argument('--save-init-lm', dest='save_init_lm', default=None, help='copy the freshly built lattice here BEFORE the run')
    ap.add_argument('--keep-lm', dest='keep_lm', default=None, help='copy the finished lattice here AFTER the run (identity gate)')
    ap.add_argument('--lattice-frames', dest='lattice_frames', type=int, default=0,
                    help='write this many lattice frames over the run (identity gate; 0 = none)')
    ap.add_argument('--state-from', dest='state_from', default=None,
                    help='seed the final chains and body snapshot of a finished gradient_growth.py npz (see the module doc)')


def active(args):
    return bool(args.bench or args.reuse_lm or args.save_init_lm or args.keep_lm)


def _dir_index(d):
    raw = (int(d[0]) + 1) + 3 * (int(d[1]) + 1) + 9 * (int(d[2]) + 1)
    return raw if raw < 13 else raw - 1


def apply(args):
    """Before build_sim: the bench clock, and the loaded state (free monomers reduced by the bound ones)."""
    tau = args.tau_us * 1e-6
    if args.bench:
        fs = args.frame_steps or max(1, int(round(args.frame_dt / tau)))
        args.t_end, fs = scene.bench_plan(args.bench, fs, tau)
        args.frame_dt = fs * tau
    args.state = None
    if args.state_from:
        d = np.load(args.state_from, allow_pickle=True)
        lens = np.asarray(d['bb_len'], np.int64); pos = np.asarray(d['bb_pos'], np.int64)
        snaps = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_body_'))
        if not snaps:
            raise SystemExit('--state-from %s: no body snapshot (run it with --snap-frames)' % args.state_from)
        idx = np.asarray(d['snap_body_%d' % snaps[-1]], np.int64)
        nx = args.nx
        bodies = [(int(v % nx), int((v // nx) % nx), int(v // (nx * nx))) for v in idx]
        off = np.concatenate([[0], np.cumsum(lens)])
        chains = []
        for i in range(len(lens)):
            p = pos[off[i]:off[i + 1]]
            chains.append((p, [_dir_index(p[j + 1] - p[j]) for j in range(len(p) - 1)]))
        bound = int(lens.sum())
        if bound > args.n_monomers:
            raise SystemExit('--state-from binds %d subunits, more than --n-monomers %d' % (bound, args.n_monomers))
        args.n_monomers -= bound
        args.state = {'chains': chains, 'bodies': bodies, 'bound': bound}
        print('[gradient] state from %s: %d chains holding %d subunits, %d bodies, %d monomers free'
              % (args.state_from, len(chains), bound, len(bodies), args.n_monomers), flush=True)


def seeded_chains(args, default):
    return args.state['chains'] if args.state else default


def bodies(args, n_bodies, default_fn):
    if not args.state:
        return default_fn()
    if len(args.state['bodies']) != n_bodies:
        raise SystemExit('--state-from holds %d bodies, the scene %d: the crowder fraction differs'
                         % (len(args.state['bodies']), n_bodies))
    return args.state['bodies']


def lattice_frames(sim, args):
    if args.lattice_frames > 0:
        sim.setLatticeWriteInterval(scene.lattice_write_steps(args.t_end, args.tau_us * 1e-6, args.lattice_frames))


def run(sim, sol, gpu, args):
    return scene.run_scene(sim, sol, gpu, reuse_lm=args.reuse_lm, save_init_lm=args.save_init_lm,
                           keep_lm=args.keep_lm, tag='gradient')


def digest(l1):
    """The identity record of one frame: sha256 of the complex-id plane (frame_hashes.py reads it as frame_hash)."""
    return hashlib.sha256(np.ascontiguousarray(l1, np.int32).tobytes()).hexdigest()
