"""Non-reacting two-species mixture in ReaDDy: the arbitration run for the purely diffusive case.

A (sigma 3) and B (sigma 6) at total packing eta = 0.20 split equally by volume, matching the
hard-sphere BD production runs (bd_two/prod_s*_r2.npz: L = 80, N_A = 3622, N_B = 453, h = 10).
Harmonic repulsion with the contact distance solved so the Barker-Henderson effective diameter
equals each sigma_ij (k = 50 kT/nm^2), the same calibration as the reacting arms. Records the
subvolume occupancy of both species on the h = 10 grid; the comparison observables are P(n_A),
P(n_B), the Fano factors, and the cross-species Pearson correlation.
"""
import argparse, json, os, sys
import numpy as np
import readdy
from readdy_rx import bh_contact

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--na', type=int, default=3622)
ap.add_argument('--nb', type=int, default=453)
ap.add_argument('--L', type=float, default=80.0)
ap.add_argument('--dt', type=float, default=1.6e-7)
ap.add_argument('--tau-max', type=float, default=4.0)
ap.add_argument('--stride', type=int, default=2000)
ap.add_argument('--k-soft', type=float, default=50.0)
ap.add_argument('--sigma-b', type=float, default=6.0)
ap.add_argument('--frac', type=float, default=0.0,
                help='if > 0, RSA slab start: all particles inserted at x < frac*L')
ap.add_argument('--out', default='out_two/rt_s{seed}.npz')
A = ap.parse_args()


def slab_init_mix(N_by, L, sig, rng, frac, tries=40000):
    """Random sequential insertion confined to x < frac*L, largest species first; periodic in
    y and z only. The same initializer as the Brownian and lattice relaxation arms, so the
    tau = 0 statistics of all descriptions coincide."""
    order = np.argsort(-np.asarray(sig))
    tot = int(sum(N_by))
    pos = np.empty((tot, 3)); spec = np.empty(tot, dtype=np.int64); n = 0
    for sp in order:
        placed, target = 0, int(N_by[sp])
        while placed < target:
            cand = rng.random((max(4096, target - placed), 3)) * L
            cand[:, 0] *= frac
            for c in cand:
                if n:
                    d = pos[:n] - c
                    d[:, 1:] -= L * np.round(d[:, 1:] / L)
                    smin = 0.5 * (sig[sp] + np.asarray(sig)[spec[:n]])
                    if ((d * d).sum(1) < smin * smin).any():
                        continue
                pos[n] = c; spec[n] = sp; n += 1; placed += 1
                if placed == target:
                    break
            tries -= 1
            if tries <= 0:
                raise RuntimeError('slab RSA stalled at %d/%d' % (placed, target))
    return pos, spec

sig = {'A': 3.0, 'B': A.sigma_b}
D = {'A': 1e5, 'B': 1e5 * 3.0 / A.sigma_b}   # Stokes scaling; equilibrium statistics do not depend on D

system = readdy.ReactionDiffusionSystem([A.L] * 3, unit_system=None)
system.kbt = 1.0
for nm in 'AB':
    system.add_species(nm, D[nm])
for i, a in enumerate('AB'):
    for b in 'AB'[i:]:
        tgt = 0.5 * (sig[a] + sig[b])
        d = bh_contact(tgt, A.k_soft)
        system.potentials.add_harmonic_repulsion(a, b, A.k_soft, d)

sim = system.simulation(kernel='CPU')
sim.output_file = ''
rng = np.random.default_rng(A.seed)
if A.frac > 0:
    pos, spec = slab_init_mix([A.na, A.nb], A.L, [sig['A'], sig['B']], rng, A.frac)
    for i, nm in enumerate('AB'):
        sim.add_particles(nm, pos[spec == i] - A.L / 2)   # box coords -> readdy centered coords
else:
    for nm, n in (('A', A.na), ('B', A.nb)):
        sim.add_particles(nm, rng.uniform(-A.L / 2, A.L / 2, size=(n, 3)))

h = 10.0
mcell = int(round(A.L / h))
occ = {'A': [], 'B': []}
def obs(x):
    types, _, poss = x
    P = np.array([[v[0], v[1], v[2]] for v in poss]); t = np.array(types)
    idx = np.floor(((P + A.L / 2) % A.L) / h).astype(np.int64) % mcell
    lin = (idx[:, 0] * mcell + idx[:, 1]) * mcell + idx[:, 2]
    for nm in 'AB':
        occ[nm].append(np.bincount(lin[t == nm], minlength=mcell ** 3).astype(np.int16))

sim.observe.particles(A.stride, callback=obs, save=False)
n_steps = int(round(A.tau_max * h * h / (D['A'] * A.dt)))
sim.record_trajectory = False
sim.run(n_steps, A.dt, show_summary=False)

tau = np.arange(len(occ['A'])) * A.stride * A.dt * D['A'] / h ** 2
meta = dict(seed=A.seed, na=A.na, nb=A.nb, L=A.L, dt=A.dt, k_soft=A.k_soft,
            stride=A.stride, tau_max=A.tau_max)
out = A.out.format(seed=A.seed); os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
np.savez_compressed(out, meta=json.dumps(meta), tau=tau,
                    occA=np.array(occ['A']), occB=np.array(occ['B']))
print('seed %d done -> %s' % (A.seed, out)); sys.stdout.flush()
