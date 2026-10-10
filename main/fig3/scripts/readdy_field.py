"""ReaDDy field arm for the lambda=8h main-figure configuration: A+B<=>C in the sinusoid,
gamma=0.3, lambda=80, L=80, koff=5e5, DetailedBalance handler, BH-calibrated soft exclusion.
Independent continuum reference for both the profile and the KL panels (soft-sphere caveat)."""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sinpot'))
import readdy, readdy_sinpot

SIG = {'A': 3.0, 'B': 3.0, 'C': 3.78}
K_SOFT = 50.0

def bh_contact(target, k):
    def d_eff(d):
        r = np.linspace(0, d, 4000)
        return np.trapz(1.0 - np.exp(-0.5*k*(r-d)**2), r)
    lo, hi = target, target + 2.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        lo, hi = (mid, hi) if d_eff(mid) < target else (lo, mid)
    return 0.5*(lo+hi)

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n0', type=int, default=1600)
ap.add_argument('--L', type=float, default=80.0)
ap.add_argument('--gamma', type=float, default=0.3)
ap.add_argument('--wavelength', type=float, default=80.0)
ap.add_argument('--K', type=float, default=200.0)
ap.add_argument('--koff', type=float, default=5.0e5)
ap.add_argument('--R-doi', type=float, default=4.5)
ap.add_argument('--D0', type=float, default=1.0e5)
ap.add_argument('--dt', type=float, default=1.6e-7)
ap.add_argument('--tau-max', type=float, default=6.0)
ap.add_argument('--stride', type=int, default=2000)
ap.add_argument('--nbin', type=int, default=16)
ap.add_argument('--out', default='out_field/rf_s{seed}.npz')
A = ap.parse_args()

sAB = 3.0
lam = A.K*A.koff/((4/3)*math.pi*(A.R_doi**3 - sAB**3))
system = readdy.ReactionDiffusionSystem([A.L]*3, unit_system=None); system.kbt = 1.0
for sname in 'ABC':
    system.add_species(sname, A.D0)
    # gamma equal per species, matching the lattice/BD arms; phase 0 with x centered at 0:
    # Phi = gamma sin(2 pi (x+L/2)/lambda) so profiles bin in [0,L) like the other arms
    system.potentials.add_custom_external(sname, readdy_sinpot.Sinusoid, A.gamma,
                                          A.wavelength, math.pi)  # sin(kx+pi) = sin(k(x+L/2)), k L = 2 pi
for i, si in SIG.items():
    for j, sj in SIG.items():
        if i <= j:
            system.potentials.add_harmonic_repulsion(i, j, K_SOFT, bh_contact(0.5*(si+sj), K_SOFT))
system.reactions.add_fusion('bind', 'A', 'B', 'C', lam, A.R_doi)
system.reactions.add_fission('unbind', 'C', 'A', 'B', A.koff, A.R_doi)
sim = system.simulation(kernel='SingleCPU')
sim.reaction_handler = 'DetailedBalance'
rng = np.random.default_rng(A.seed)
sim.add_particles('A', rng.random((A.n0, 3))*A.L - A.L/2)
sim.add_particles('B', rng.random((A.n0, 3))*A.L - A.L/2)

h = 10.0
mcell = int(round(A.L / h))
prof = {k: [] for k in 'ABC'}; counts = []; occ = {'A': [], 'C': []}
def obs(x):
    # the particles observable delivers TYPE NAMES (strings), not ids
    types, _, poss = x
    P = np.array([[v[0], v[1], v[2]] for v in poss]); t = np.array(types)
    px = P[:, 0]
    counts.append([int((t == nm).sum()) for nm in 'ABC'])
    for nm in 'ABC':
        prof[nm].append(np.histogram((px[t == nm] + A.L/2) % A.L, bins=A.nbin, range=(0, A.L))[0])
    # subvolume occupancy on the same h-grid as the lattice and BD, for the KL comparison
    idx = np.floor(((P + A.L/2) % A.L) / h).astype(np.int64) % mcell
    lin = (idx[:, 0] * mcell + idx[:, 1]) * mcell + idx[:, 2]
    for nm in 'AC':
        occ[nm].append(np.bincount(lin[t == nm], minlength=mcell**3).astype(np.int16))
stride = A.stride
sim.observe.particles(stride=stride, callback=obs, save=False)
steps = int(math.ceil(A.tau_max*h*h/A.D0/A.dt))
sim.show_progress = False
sim.run(steps, A.dt, show_summary=False)
tau = np.arange(len(counts))*stride*A.dt*A.D0/(h*h)
meta = dict(seed=A.seed, n0=A.n0, L=A.L, gamma=A.gamma, wavelength=A.wavelength, K=A.K,
            koff=A.koff, R_doi=A.R_doi, lam=lam, dt=A.dt, steps=steps, k_soft=K_SOFT,
            nbin=A.nbin, readdy=readdy.__version__)
out = A.out.format(seed=A.seed); os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
np.savez_compressed(out, meta=json.dumps(meta), tau=tau, counts=np.array(counts),
                    profA=np.array(prof['A']), profB=np.array(prof['B']), profC=np.array(prof['C']),
                    occA=np.array(occ['A']), occC=np.array(occ['C']))
print('seed %d done -> %s' % (A.seed, out)); sys.stdout.flush()
