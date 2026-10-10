"""ReaDDy reference for the reversible bimolecular test: A + B <=> C with soft exclusion
calibrated to the hard-sphere diameters by the Barker-Henderson effective diameter.

Third-party closure of the FN loop: the lattice reaction kernel gates with the Frohner-Noe
acceptance; ReaDDy's DetailedBalance handler is the same construction in continuum, by the same
authors. lambda is passed UNCORRECTED (no 0.906): the DB handler is built to reach the interacting
equilibrium exactly, so agreement with the reaction-ensemble value here also tests whether our
kinetic sampler's lambda calibration was compensating for physics ReaDDy carries by construction.

Unitless kBT=1; lengths in nm, times in s (D = 1e5 nm^2/s matches every other arm).
"""
import argparse, json, math, os, sys
import numpy as np
import readdy

SIG = {'A': 3.0, 'B': 3.0, 'C': 3.78}
K_SOFT = 50.0            # kT/nm^2 harmonic stiffness; contact distance solved per pair below


def bh_contact(target, k):
    """Contact distance d such that the Barker-Henderson effective diameter of the harmonic
    repulsion U = k/2 (r-d)^2 (r<d) equals the hard-sphere target."""
    def d_eff(d):
        r = np.linspace(0, d, 4000)
        return np.trapz(1.0 - np.exp(-0.5 * k * (r - d) ** 2), r)
    lo, hi = target, target + 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if d_eff(mid) < target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n0', type=int, default=200)
    ap.add_argument('--L', type=float, default=40.0)
    ap.add_argument('--K', type=float, default=200.0)
    ap.add_argument('--koff', type=float, default=5.0e3)
    ap.add_argument('--R-doi', type=float, default=4.5)
    ap.add_argument('--D0', type=float, default=1.0e5)
    ap.add_argument('--dt', type=float, default=1.6e-7)
    ap.add_argument('--tau-max', type=float, default=4.0, help='in units of D t / h^2, h=10')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--soft', action='store_true', help='skip exclusion (ideal-gas control)')
    ap.add_argument('--out', default='out/rd_s{seed}.npz')
    A = ap.parse_args()

    sAB = 0.5 * (SIG['A'] + SIG['B'])
    lam = A.K * A.koff / ((4/3) * math.pi * (A.R_doi**3 - sAB**3))   # UNCORRECTED

    system = readdy.ReactionDiffusionSystem([A.L]*3, unit_system=None)
    system.kbt = 1.0
    for s in 'ABC':
        system.add_species(s, A.D0)
    if not A.soft:
        for i, si in SIG.items():
            for j, sj in SIG.items():
                if i <= j:
                    d = bh_contact(0.5*(si+sj), K_SOFT)
                    system.potentials.add_harmonic_repulsion(i, j, K_SOFT, d)
    system.reactions.add_fusion('bind', 'A', 'B', 'C', lam, A.R_doi)
    system.reactions.add_fission('unbind', 'C', 'A', 'B', A.koff, A.R_doi)

    sim = system.simulation(kernel='SingleCPU')
    sim.reaction_handler = 'DetailedBalance'
    rng = np.random.default_rng(A.seed)
    sim.add_particles('A', rng.random((A.n0, 3)) * A.L - A.L/2)
    sim.add_particles('B', rng.random((A.n0, 3)) * A.L - A.L/2)
    counts = {'t': [], 'n': []}
    def obs(c):
        counts['n'].append([int(x) for x in c])
    sim.observe.number_of_particles(stride=200, types=['A', 'B', 'C'], callback=obs, save=False)

    h = 10.0
    steps = int(math.ceil(A.tau_max * h*h / A.D0 / A.dt))
    sim.output_file = ''
    sim.show_progress = False
    sim.run(steps, A.dt, show_summary=False)

    n = np.array(counts['n'])
    tau = np.arange(len(n)) * 200 * A.dt * A.D0 / (h*h)
    k = tau >= tau[-1]/2
    NC = n[k, 2].mean()
    V = A.L**3
    K_eff = NC * V / (n[k,0].mean() * n[k,1].mean())
    meta = dict(seed=A.seed, n0=A.n0, L=A.L, K=A.K, koff=A.koff, R_doi=A.R_doi, lam=lam,
                dt=A.dt, steps=steps, soft_excluded=not A.soft, k_soft=K_SOFT,
                NC_tail=float(NC), K_eff=float(K_eff), K_eff_over_K=float(K_eff/A.K),
                readdy=readdy.__version__)
    out = A.out.format(seed=A.seed)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), tau=tau, counts=n)
    print('seed %d: N_C tail %.1f  K_eff/K %.3f -> %s' % (A.seed, NC, K_eff/A.K, out))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
