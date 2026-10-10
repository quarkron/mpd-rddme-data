"""Write a scene directory for the single-core port (lmcpu).

  make_scene.py ternary <dir> [--dims 32,8,8] [--seed 0]
      the ternary cost scene of adv_ternary.py: 34225 A (2 nm, 0.1 um^2/s), 4278 B (4 nm, 0.05),
      type 2 the 5 nm FILL placeholder, four 20 nm bodies at x in {4,12,20,28}, y = z = 4, EXACT hop,
      free-volume functional, reciprocal fill counted as A, smear incremental with a full re-stamp
      every 512 steps, Q = 8, tau 50 us.
  make_scene.py demix <dir> [--nx 64] [--state-from <lattice npz>] [--seed 1]
      the polymer demixing scene of gradient_growth.py (studies/polymer) as the cost campaign ran it:
      60000 monomers (10 nm, 0.3 um^2/s), 20 nm bodies at phi 0.20 (0.04 um^2/s, Q = 2), a 3 nm bath at
      eta 0.05 (0.15 um^2/s), LINEAR hop, smear folded into the bath densities and re-stamped every
      step, nucleation 0.02 /s, growth 20 /s, shrink 3 /s, kappa 4, stretch 400 about sqrt(2), four
      interior attempts of the SI move set (end-bond rotation, full kink, edge translation; no crankshaft,
      no rigid rotation), the face-vector collective hop, tau 20 us. --state-from seeds the final chains bead for
      bead and its last body snapshot, as the ReaDDy cost arm was seeded (adv_demix_scene.py).

Every number is the one the GPU driver passes, cited beside it; the stencil is the GPU's own
ribosome_blob.fmt_stencil. Nothing here is fitted.
"""
import argparse, math, os, sys

import numpy as np

ENGINE = os.environ.get('LM_ENGINE_DIR', '/home/racda/ftsz_tests/Lattice_Microbes_2.6/testing/engine')
sys.path.insert(0, ENGINE)
import ribosome_blob as rb   # noqa: E402


def write(dirname, params, species, arrays):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, 'params.txt'), 'w') as f:
        for k, v in params.items():
            f.write('%s %s\n' % (k, v))
        for t, (sig, D, n) in sorted(species.items()):
            f.write('species %d %.9g %.9g %d\n' % (t, sig, D, n))
    for name, arr in arrays.items():
        arr.tofile(os.path.join(dirname, name))
    print('[make_scene] %s: %s' % (dirname, ', '.join('%s=%s' % kv for kv in params.items())))


def count(eta, sigma, xyz, V):
    return int(round(eta * xyz * V / ((math.pi / 6.0) * sigma ** 3)))   # bath_species.count


def ternary(A):
    NX, NY, NZ = (int(v) for v in A.dims.split(','))
    h, tau = 10.0, 50e-6                     # adv_ternary NM, TAU_S
    V, xyz = h ** 3, NX * NY * NZ
    sA, sB, sC, sFill = 2.0, 4.0, 20.0, 5.0  # adv_ternary sigma_a/b/c, SIGMA_FILL
    D_A = 0.1e6                              # D_A_UM2 in nm^2/s
    D_B = D_A * sA / sB                      # D_UM2['B'] = D_A sigma_A / sigma_B
    D_C = D_A * sA / sC                      # D_UM2['C']
    per = 8                                  # PER: one body per (80 nm)^3
    centers = [(per * i + per // 2, per * j + per // 2, per * k + per // 2)
               for i in range(max(1, NX // per)) for j in range(max(1, NY // per)) for k in range(max(1, NZ // per))]
    params = dict(NX=NX, NY=NY, NZ=NZ, h=h, tau=tau, nt=3,
                  lambda_=1.0, exactHop=1, freeVol=1, reciprocal=1, fillSpecies=1,
                  bodySpecies=2, smearIncremental=512, Q=8,
                  bodyPdir=D_C * tau / h ** 2,  # setStructureSizeDiffusion x HopSpeciesScale = D_C tau / h^2
                  bodyPair=1, beadBody=1, friction=1, subunitSigmaNm=sFill,
                  chains=0, seed=A.seed, steps1=300, steps2=1500, frameEvery=0)
    params['lambda'] = params.pop('lambda_')
    species = {1: (sA, D_A, count(0.07, sA, xyz, V)),
               2: (sFill, D_A, 0),                    # FILL: region default D, no particles
               3: (sB, D_B, count(0.07, sB, xyz, V))}
    if A.no_bodies:                          # checkpoint against adv_cpu.cpp: the bath alone, same box
        centers = []
        params.update(freeVol=0)
    arrays = {'stencil.f32': rb.fmt_stencil(sC, h).reshape(-1).astype(np.float32),
              'bodies.f64': np.array(centers, np.float64).reshape(-1)}
    write(A.dir, params, species, arrays)


def place_bodies(n, nx, rng, min_sep=2):
    """gradient_growth.place_bodies on a flat profile: uniform voxels, no two within min_sep on any axis."""
    taken = np.zeros((nx, nx, nx), bool)
    pts, guard = [], 0
    while len(pts) < n and guard < 200 * n + 100000:
        guard += 1
        v = int(rng.integers(nx ** 3))
        z, rem = divmod(v, nx * nx); y, x = divmod(rem, nx)
        lo = [max(0, c - (min_sep - 1)) for c in (x, y, z)]; hi = [min(nx, c + min_sep) for c in (x, y, z)]
        if taken[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
            continue
        taken[x, y, z] = True; pts.append((x, y, z))
    if len(pts) < n:
        raise SystemExit('placed %d/%d bodies' % (len(pts), n))
    return pts


def demix(A):
    nx, h, tau = A.nx, 10.0, 20e-6
    V, xyz = h ** 3, nx ** 3
    sM, sR, sB, sC = 10.0, 5.0, 3.0, 20.0       # --sigma, SIGMA_BODY_ROW, --bath-sigma, the body
    D_M, D_R, D_B, D_C = 0.3e6, 0.04e6, 0.15e6, 0.04e6
    n_c = int(round(A.phi * xyz * V / ((math.pi / 6.0) * sC ** 3)))
    n_bath = count(A.bath_eta, sB, xyz, V)
    arrays = {'stencil.f32': rb.fmt_stencil(sC, h).reshape(-1).astype(np.float32)}
    bound = 0
    if A.state_from:
        d = np.load(A.state_from, allow_pickle=True)
        lens = np.asarray(d['bb_len'], np.int32); pos = np.asarray(d['bb_pos'], np.int32)
        snaps = sorted(int(k.split('_')[-1]) for k in d.files if k.startswith('snap_body_'))
        if not snaps:
            raise SystemExit('%s has no body snapshot' % A.state_from)
        idx = np.asarray(d['snap_body_%d' % snaps[-1]], np.int64)
        bodies = np.stack([idx % nx, (idx // nx) % nx, idx // (nx * nx)], 1).astype(np.float64)
        if len(bodies) != n_c:
            raise SystemExit('state holds %d bodies, the scene %d' % (len(bodies), n_c))
        arrays.update({'chain_len.i32': lens, 'chain_vox.i32': pos.reshape(-1)})
        bound = int(lens.sum())
        print('[make_scene] state %s: %d chains, %d bound, %d bodies' % (A.state_from, len(lens), bound, len(bodies)))
    else:
        bodies = np.array(place_bodies(n_c, nx, np.random.default_rng(A.seed + 7919)), np.float64)
    arrays['bodies.f64'] = bodies.reshape(-1)
    params = dict(NX=nx, NY=nx, NZ=nx, h=h, tau=tau, nt=3,
                  exactHop=0, freeVol=0, reciprocal=1, fillSpecies=1,  # setHopIntegration never called: LINEAR
                  bodySpecies=2, smearIncremental=0, Q=2,              # incremental never armed; subhop-q 2
                  bodyPdir=D_C * tau / h ** 2,
                  bodyPair=1, beadBody=1, friction=1, subunitSigmaNm=sM,
                  chains=1, kNuc=0.02, kPoly=20.0, kDepoly=3.0, kappa=4.0, kStretch=400.0, bond0=math.sqrt(2.0),
                  interiorAttempts=int(round(10 * 20 / 50)),          # ARMS: round(10 tau_us / 50)
                  monomerSpecies=1, chainD1=D_M * tau / h ** 2,
                  # --si-moves: no rigid rotation, interior mask 0x1B (the port's default)
                  seed=A.seed, steps1=50, steps2=250, frameEvery=0)
    params['lambda'] = 1.0
    species = {1: (sM, D_M, A.n_monomers - bound), 2: (sR, D_R, 0), 3: (sB, D_B, n_bath)}
    write(A.dir, params, species, arrays)


def statics(A):
    """Fixed-length chains, no bath, no bodies, no chemistry: the move set alone, against the exact
    self-avoiding FCC chain with bending (statics_len.npz r2_exact_saw). Chains seeded straight along
    (1,1,0), one per box and centred (the reference is an isolated chain; independent boxes are
    independent seeds), the harness's 200 interior attempts per step."""
    n, nx, h, tau = A.n, A.box, 10.0, 20e-6
    if A.chains != 1:
        raise SystemExit('statics seeds ONE chain per box: the reference is an isolated chain')
    c0 = (nx - n) // 2                                   # centred, clear of every wall
    lens = [n]
    vox = [(c0 + i, c0 + i, nx // 2) for i in range(n)]  # (1,1,0) edge bonds, parity class fixed
    params = dict(NX=nx, NY=nx, NZ=nx, h=h, tau=tau, nt=3, exactHop=0, freeVol=0, reciprocal=1, fillSpecies=1,
                  bodySpecies=2, smearIncremental=0, Q=2, bodyPdir=0.0, bodyPair=1, beadBody=1, friction=1,
                  subunitSigmaNm=10.0, chains=1, kNuc=0.0, kPoly=0.0, kDepoly=0.0, kappa=A.kappa,
                  kStretch=400.0, bond0=math.sqrt(2.0), interiorAttempts=200, monomerSpecies=1,
                  chainD1=0.06,
                  seed=A.seed, steps1=0, steps2=2000, frameEvery=10)
    params['lambda'] = 1.0
    species = {1: (10.0, 0.3e6, 0), 2: (5.0, 0.04e6, 0), 3: (3.0, 0.15e6, 0)}
    arrays = {'stencil.f32': rb.fmt_stencil(20.0, h).reshape(-1).astype(np.float32),
              'bodies.f64': np.zeros(0, np.float64),
              'chain_len.i32': np.array(lens, np.int32), 'chain_vox.i32': np.array(vox, np.int32).reshape(-1)}
    write(A.dir, params, species, arrays)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='scene', required=True)
    t = sub.add_parser('ternary')
    t.add_argument('dir'); t.add_argument('--dims', default='32,8,8'); t.add_argument('--seed', type=int, default=0)
    t.add_argument('--no-bodies', dest='no_bodies', action='store_true')
    d = sub.add_parser('demix')
    d.add_argument('dir'); d.add_argument('--nx', type=int, default=64); d.add_argument('--seed', type=int, default=1)
    d.add_argument('--n-monomers', dest='n_monomers', type=int, default=60000)
    d.add_argument('--phi', type=float, default=0.20); d.add_argument('--bath-eta', dest='bath_eta', type=float, default=0.05)
    d.add_argument('--state-from', dest='state_from', default=None)
    q = sub.add_parser('statics')
    q.add_argument('dir'); q.add_argument('--n', type=int, default=16); q.add_argument('--kappa', type=float, default=4.0)
    q.add_argument('--box', type=int, default=37); q.add_argument('--chains', type=int, default=1)
    q.add_argument('--seed', type=int, default=1)
    A = ap.parse_args()
    {'ternary': ternary, 'demix': demix, 'statics': statics}[A.scene](A)


if __name__ == '__main__':
    main()
