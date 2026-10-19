"""Non-reacting ternary mixture in ReaDDy: two small hard-sphere species and one body larger than the lattice
voxel, the reference for the manuscript's oversize-species relaxation figure.

A (sigma 2 nm) and B (sigma 4 nm) at packing eta_A = eta_B = 0.07, and C (sigma 20 nm) as a dilute tracer, one per
(80 nm)^3 box (0.8 % by volume; the 32^3 lattice arm at the same number density carries 64). Harmonic repulsion with
the contact distance solved so the Barker-Henderson effective diameter equals each sigma_ij (k = 50 kT/nm^2), the
calibration every other ReaDDy arm in the manuscript uses. Records the subvolume occupancy of A and B on the
h = 10 nm grid and C's position at every frame, so the analysis can bin the bath around C in C's own frame.

TWO THINGS THAT DECIDE THE COST, measured 2026-09-07 at this state point, 16 threads, 80 nm box:

  The C-C potential. ReaDDy keeps one neighbour list with one cutoff, the largest of all potentials. A C-C
  repulsion puts that cutoff at 20 nm and every A-A search then spans a 20 nm sphere holding thousands of
  particles: 1237 ms per step, 515 min per seed. With ONE C per box there is no C-C pair for it to act on, so
  it is omitted (28 ms per step, 12 min per seed). More than one C therefore needs --cc, and pays for it.

  C mobile against C static. As a one-body sphere (--c-mode static) C leaves the neighbour list altogether
  (3.2 ms per step, 1.3 min per seed). C is ten times slower than A by Stokes scaling and its mobility is already
  validated against ReaDDy by the friction ladder, so the static run is a legitimate control for the bath's
  statistics around C; the mobile run is the complete reference and is cheap enough to be the default.

The lattice side caps a voxel at 64 particles, and the slab start DOUBLES the local density: at sigma_A = 2 nm and
h = 10 nm, eta_A = eta_B = 0.10 puts 54 particles in a filled voxel on average and 79 at the tail (measured over
three seeds), which the lattice driver refuses. At 0.07 / 0.07, the largest equal-volume split the start allows,
the mean is 38 and the tail 55. That is why the defaults are 0.07 and not the binary figure's 0.10.

Initial state: A and B by random sequential insertion into x < frac*L, periodic in y and z, the same initializer as
the binary arms so the tau = 0 statistics of every description coincide; C at the centre of the EMPTY half by
default (--c-init), so the depletion shell forms as the bath arrives. The clock is tau = D_A t / h^2 with
D_A = 1e5 nm^2/s, D_B and D_C by Stokes scaling, dt = 1.6e-7 s (D_A k dt = 0.8, the validated step).
"""
import argparse, json, os, sys, time
import numpy as np
import readdy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readdy_rx import bh_contact

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--sigma-a', type=float, default=2.0)
ap.add_argument('--sigma-b', type=float, default=4.0)
ap.add_argument('--sigma-c', type=float, default=20.0)
ap.add_argument('--eta-a', type=float, default=0.07)
ap.add_argument('--eta-b', type=float, default=0.07)
ap.add_argument('--n-c', type=int, default=1, help='C particles per box; > 1 needs --cc (see the docstring)')
ap.add_argument('--cc', action='store_true', help='include the C-C repulsion (sets the neighbour cutoff to sigma_C: ~40x slower)')
ap.add_argument('--c-mode', choices=('mobile', 'static'), default='mobile',
                help='mobile: C is a particle (default); static: C is a fixed one-body exclusion sphere')
ap.add_argument('--c-init', choices=('empty-centre', 'filled-centre', 'box-centre', 'uniform'), default='empty-centre',
                help='where C starts; the lattice arm must use the same voxel')
ap.add_argument('--L', type=float, default=80.0)
ap.add_argument('--h', type=float, default=10.0)
ap.add_argument('--dt', type=float, default=1.6e-7)
ap.add_argument('--k-soft', type=float, default=50.0)
ap.add_argument('--d-a', type=float, default=1e5, help='D_A in nm^2/s; the clock tau = D_A t / h^2')
ap.add_argument('--tau-max', type=float, default=4.0)
ap.add_argument('--stride', type=int, default=500)
ap.add_argument('--frac', type=float, default=0.5, help='RSA slab: A and B inserted at x < frac*L (0 = uniform)')
ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--probe', type=int, default=0, help='time this many steps, print the extrapolation, exit')
ap.add_argument('--gr', action='store_true', help='accumulate the radial distribution of A and B around C over frames with tau >= --tau-eq')
ap.add_argument('--tau-eq', type=float, default=2.0)
ap.add_argument('--gr-dr', type=float, default=0.25, help='nm bin width of the radial histogram (0 .. L/2)')
ap.add_argument('--out', default='out_three_relax/rt3_s{seed}.npz')
A = ap.parse_args()
if A.n_c > 1 and not A.cc:
    sys.exit('n_c > 1 without --cc: two C particles would overlap freely; add --cc (and pay the 20 nm cutoff)')
if A.c_mode == 'static' and A.n_c != 1:
    sys.exit('static C is one sphere at one place; use --n-c 1')

sig = {'A': A.sigma_a, 'B': A.sigma_b, 'C': A.sigma_c}
D = {'A': A.d_a, 'B': A.d_a * A.sigma_a / A.sigma_b, 'C': A.d_a * A.sigma_a / A.sigma_c}   # Stokes scaling
L, h = A.L, A.h
V = L ** 3
N = {'A': int(round(A.eta_a * V / (np.pi / 6 * sig['A'] ** 3))),
     'B': int(round(A.eta_b * V / (np.pi / 6 * sig['B'] ** 3))),
     'C': A.n_c}
rng = np.random.default_rng(A.seed)


def c_positions():
    """C's initial position(s) in box coordinates [0, L)^3."""
    if A.c_init == 'uniform':
        return rng.uniform(0, L, size=(N['C'], 3))
    x = {'empty-centre': (1 + A.frac) / 2 * L, 'filled-centre': A.frac / 2 * L, 'box-centre': L / 2}[A.c_init]
    return np.array([[x, L / 2, L / 2]] * N['C'])


def slab_init_mix(N_by, sig_by, frac, cpos, tries=40000):
    """Random sequential insertion of the bath, largest species first, confined to x < frac*L (frac = 0 means the
    whole box), periodic in y and z, rejecting overlap with each other AND with the C sphere(s), which sit at cpos
    from the start. The same initializer as the binary arms.

    Search by cell list rather than against every placed particle: the quadratic version cost ~5 min per seed at
    13.7k particles. Candidates are drawn and tested in the identical order with the identical rejection rule, so
    the accepted configuration is bit-identical to the quadratic version for the same seed (checked at build time)."""
    frac = frac if frac > 0 else 1.0
    order = np.argsort(-np.asarray(sig_by))
    tot = int(sum(N_by))
    pos = np.empty((tot, 3)); spec = np.empty(tot, dtype=np.int64); n = 0
    cell = max(sig_by)                                   # >= every bath-bath contact distance
    nc = max(1, int(L // cell)); cell = L / nc
    grid = {}                                            # cell index (ix,iy,iz) -> list of particle indices
    def key(p):
        return tuple((np.floor(p / cell).astype(int)) % nc)
    def near(c):
        kx, ky, kz = key(c)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(grid.get(((kx + dx) % nc, (ky + dy) % nc, (kz + dz) % nc), ()))
        return out
    for sp in order:
        placed, target = 0, int(N_by[sp])
        rc = 0.5 * (sig_by[sp] + sig['C'])
        while placed < target:
            cand = rng.random((max(4096, target - placed), 3)) * L
            cand[:, 0] *= frac
            for c in cand:
                dC = cpos - c
                dC -= L * np.round(dC / L)
                if ((dC * dC).sum(1) < rc * rc).any():
                    continue
                nb = near(c)
                if nb:
                    d = pos[nb] - c
                    d[:, 1:] -= L * np.round(d[:, 1:] / L)      # periodic in y, z only, as the slab is
                    smin = 0.5 * (sig_by[sp] + np.asarray(sig_by)[spec[nb]])
                    if ((d * d).sum(1) < smin * smin).any():
                        continue
                pos[n] = c; spec[n] = sp
                grid.setdefault(key(c), []).append(n)
                n += 1; placed += 1
                if placed == target:
                    break
            tries -= 1
            if tries <= 0:
                raise RuntimeError('slab RSA stalled at %d/%d of species %d' % (placed, target, sp))
    return pos, spec


system = readdy.ReactionDiffusionSystem([L] * 3, unit_system=None)
system.kbt = 1.0
bath = ['A', 'B']
for nm in bath:
    system.add_species(nm, D[nm])
for i, a in enumerate(bath):
    for b in bath[i:]:
        system.potentials.add_harmonic_repulsion(a, b, A.k_soft, bh_contact(0.5 * (sig[a] + sig[b]), A.k_soft))
cpos = c_positions()
if A.c_mode == 'mobile':
    system.add_species('C', D['C'])
    for a in bath:
        system.potentials.add_harmonic_repulsion('C', a, A.k_soft, bh_contact(0.5 * (sig['C'] + sig[a]), A.k_soft))
    if A.cc:
        system.potentials.add_harmonic_repulsion('C', 'C', A.k_soft, bh_contact(sig['C'], A.k_soft))
else:
    # The same radial harmonic as a pair potential, so the same Barker-Henderson contact applies; as a one-body
    # potential it does not enter the neighbour list.
    for a in bath:
        system.potentials.add_sphere(a, A.k_soft, list(cpos[0] - L / 2), bh_contact(0.5 * (sig['C'] + sig[a]), A.k_soft),
                                     inclusion=False)

sim = system.simulation(kernel='CPU')
sim.output_file = ''
if A.threads:
    sim.kernel_configuration.n_threads = A.threads
pos, spec = slab_init_mix([N['A'], N['B']], [sig['A'], sig['B']], A.frac, cpos)
for i, nm in enumerate(bath):
    sim.add_particles(nm, pos[spec == i] - L / 2)             # box coords -> readdy centred coords
if A.c_mode == 'mobile':
    sim.add_particles('C', cpos - L / 2)

mcell = int(round(L / h))
occ = {'A': [], 'B': []}
posC = []
gr_edges = np.arange(0.0, L / 2 + A.gr_dr, A.gr_dr); gr = {'A': np.zeros(len(gr_edges) - 1), 'B': np.zeros(len(gr_edges) - 1)}; gr_frames = [0]
n_obs = [0]
def obs(x):
    types, _, poss = x
    P = np.array([[v[0], v[1], v[2]] for v in poss]); t = np.array(types)
    if A.gr:
        tau_now = n_obs[0] * A.stride * A.dt * D['A'] / h ** 2; n_obs[0] += 1
        if tau_now >= A.tau_eq:
            cs = P[t == 'C'] if A.c_mode == 'mobile' else (cpos - L / 2)
            for c in cs:                                   # distances from C's centre, minimum image, per bath species
                for nm in bath:
                    d = P[t == nm] - c; d -= L * np.round(d / L)
                    gr[nm] += np.histogram(np.sqrt((d * d).sum(1)), bins=gr_edges)[0]
            gr_frames[0] += 1
    idx = np.floor(((P + L / 2) % L) / h).astype(np.int64) % mcell
    lin = (idx[:, 0] * mcell + idx[:, 1]) * mcell + idx[:, 2]
    for nm in bath:
        occ[nm].append(np.bincount(lin[t == nm], minlength=mcell ** 3).astype(np.int16))
    posC.append(((P[t == 'C'] + L / 2) % L) if A.c_mode == 'mobile' else cpos.copy())   # box coordinates

n_steps = int(round(A.tau_max * h * h / (D['A'] * A.dt)))
print(f"[readdy_three] N={N} L={L:.0f} h={h:.0f} sig={sig} D={ {k: f'{v:.3g}' for k, v in D.items()} } "
      f"C {A.c_mode} at {A.c_init} {cpos[0].round(1).tolist()}  cc={A.cc}  threads={sim.kernel_configuration.n_threads}", flush=True)
if A.probe:
    sim.record_trajectory = False
    t0 = time.perf_counter(); sim.run(A.probe, A.dt, show_summary=False); per = (time.perf_counter() - t0) / A.probe
    print(f"[readdy_three] probe: {per*1e3:.2f} ms/step -> {per*n_steps/60:.1f} min per seed to tau {A.tau_max} ({n_steps} steps)")
    sys.exit(0)

sim.observe.particles(A.stride, callback=obs, save=False)
sim.record_trajectory = False
t0 = time.perf_counter()
sim.run(n_steps, A.dt, show_summary=False)
wall = time.perf_counter() - t0

tau = np.arange(len(occ['A'])) * A.stride * A.dt * D['A'] / h ** 2
meta = dict(seed=A.seed, N=N, L=L, h=h, dt=A.dt, k_soft=A.k_soft, stride=A.stride, tau_max=A.tau_max,
            sigma=sig, D=D, eta_a=A.eta_a, eta_b=A.eta_b, frac=A.frac, c_mode=A.c_mode, c_init=A.c_init, cc=A.cc,
            c_init_pos=cpos.tolist(), threads=A.threads, wall_s=wall, ms_per_step=1e3 * wall / n_steps)
out = A.out.format(seed=A.seed); os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
extra = {}
if A.gr:
    # g(r) = counts / (frames * n_C * rho_s * shell volume), rho_s the box-average number density of species s; the
    # far-field plateau is then ~1 up to the volume C excludes, and the analysis renormalises on the plateau anyway.
    shell = 4.0 / 3.0 * np.pi * (gr_edges[1:] ** 3 - gr_edges[:-1] ** 3)
    for nm in bath:
        extra['gr' + nm] = gr[nm] / max(gr_frames[0], 1) / N['C'] / (N[nm] / V) / shell
        extra['grcount' + nm] = gr[nm]
    extra['gr_edges'] = gr_edges; extra['gr_frames'] = gr_frames[0]
np.savez_compressed(out, meta=json.dumps(meta), tau=tau, occA=np.array(occ['A']), occB=np.array(occ['B']),
                    posC=np.array(posC), **extra)
print(f"[readdy_three] seed {A.seed} done in {wall:.0f} s ({1e3*wall/n_steps:.1f} ms/step), {len(tau)} frames -> {out}", flush=True)
