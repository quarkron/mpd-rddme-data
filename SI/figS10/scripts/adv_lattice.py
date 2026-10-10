"""Lattice (GPU) timing and ceiling arm of the advantage-metric campaign. Runs in the lm2.6 container.

Measures three things per point, which is the whole reason this driver exists rather than reusing
vex_reactive.py:

  cost      per-step wall time, from a two-point difference over one solver run pair.
  tau_samp  the TRANSPORT ceiling, read from the adaptive-tau controller. That controller holds
            the lattice-wide maximum of the per-pass hop propensity at eps, so the converged
            multiplier tau_scale inverts directly to the extremal Bernoulli factor and hence to
            the ceiling at a declared criterion. getLastSumMax() is recorded alongside so the
            inversion can be checked rather than trusted.
  tau_rxn   the REACTION ceiling, computed from the measured per-voxel occupancy field and the
            scheme's rate table. Recorded as the max and mean total propensity in 1/s, so the
            ceiling at ANY other rate scale follows by proportionality without a new run.

The occupancy histogram is written out too: it is the input to the a priori extreme-value estimate
of maxB that the analysis tests the measurement against.
"""
import argparse, json, math, os, sys, tempfile, time
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Build resolution lives in the engine tree, not in studies/: testing/lm_build.py. Inserting a
# guessed path instead of using it is what once loaded an eight-week-old binary for a whole
# campaign, so the path is explicit and the import is not made optional.
sys.path.insert(0, os.environ.get('LM_TESTING_DIR', '/lm/testing'))
from lm_build import resolve_lm_build, report_imported_lm
resolve_lm_build()
import lm
report_imported_lm()
import adv_schemes as S

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', required=True, choices=sorted(S.SCHEMES))
ap.add_argument('--eta', type=float, required=True)
ap.add_argument('--steps1', type=int, default=2000)
ap.add_argument('--steps2', type=int, default=10000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--dt', type=float, default=S.DT_S, help='nominal step of the COST runs')
ap.add_argument('--dt-ceiling', type=float, default=2e-4,
                help='nominal step of the CEILING run, large enough that the controller engages')
ap.add_argument('--eps', type=float, default=0.2, help='adaptive-tau budget defining the ceiling')
ap.add_argument('--no-exclusion', action='store_true')
ap.add_argument('--no-field', action='store_true')
ap.add_argument('--no-adaptive', action='store_true', help='freeze tau_scale at 1 (cost-only run)')
ap.add_argument('--out', required=True)
A = ap.parse_args()

spec = S.SCHEMES[A.scheme]
names = spec['species']
sig = [spec['sigma'][n] for n in names]
Dif = S.diffusion(A.scheme)
NX, NY, NZ = S.DIMS
M = S.M_SUB
counts0 = S.counts_for_eta(A.scheme, A.eta)


def build(path, n_steps, n_frames, dt):
    import random as _random
    _random.seed(int(A.seed)); np.random.seed(int(A.seed) & 0x7FFFFFFF)
    from pyLM import RDME
    from pyLM.units import nm, micron, microsecond
    ux, uy, uz = (d * S.H_NM * 1e-3 for d in S.DIMS)
    sim = RDME.RDMESimulation(dimensions=micron(ux, uy, uz), spacing=nm(S.H_NM))
    sim.defineSpecies(list(names))
    reg = sim.modifyRegion('default')
    reg.setDefaultDiffusionRate(S.D_NM2S * 1e-18)          # nm^2/s -> m^2/s
    for n in names:
        reg.setDiffusionRate(species=n, rate=Dif[n] * 1e-18)
    for kind, react, prod, rate in S.rates(A.scheme, 'pylm'):
        if kind == 'fusion':
            reg.addReaction(reactant=tuple(react), product=prod[0], rate=rate)
        elif kind == 'fission':
            reg.addReaction(reactant=react[0], product=tuple(prod), rate=rate)
        else:
            reg.addReaction(reactant=react[0], product=prod[0], rate=rate)
    for n, c in counts0.items():
        if c:
            sim.addParticles(species=n, region='default', count=int(c))
    sim.setTimestep(microsecond(dt * 1e6))
    per_frame = max(1, n_steps // n_frames)
    sim.setWriteInterval(per_frame)
    sim.setLatticeWriteInterval(per_frame)
    sim.setSimulationTime(n_steps * dt)
    sim.setRandomSeed(int(A.seed))
    sim.save(path)
    f = lm.SimulationFile(path)
    f.setParameter('seed', str(int(A.seed)))
    f.close()
    return per_frame


def configure(solver, adaptive):
    if A.no_field:
        line = np.zeros(NX, np.float32)
    else:
        line = np.sin(2.0 * np.pi * np.arange(NX, dtype=np.float32)
                      * S.H_NM / S.WAVELENGTH).astype(np.float32)
    psi0 = np.broadcast_to(line, (NZ * NY, NX)).astype(np.float32).ravel(order='C')
    solver.setPotentialBasis(np.ascontiguousarray(psi0), np.zeros(M, np.float32))
    for i in range(len(names)):
        solver.setSpeciesCoupling(i, 0.0 if A.no_field else float(S.GAMMA), 0.0)
    if not A.no_exclusion:
        solver.setSpeciesHardSphereDiameters(np.array(sig, np.float64), float(S.V_SUB))
        solver.setHardCoreExclusion(1.0, 2)
    solver.setHopIntegration(1)
    solver.setLatticeWordBound(1)
    solver.setLatticeTransportBound(1)
    solver.setDriftSpeciesMemo(1)
    if adaptive:
        solver.setAdaptiveTau(1, float(A.eps), 0.05)
    solver.validateStability()


def run_once(steps, dt, adaptive):
    """One solver run. Returns wall time, converged tau multiplier, last sum_max, occupancy field.

    ★THE TRAP THIS SIGNATURE EXISTS FOR. lm.runSolver advances to a simulated TIME, not for a step
    count: the .lm file carries n_steps*dt seconds. With the adaptive controller engaged each step
    advances only tau_scale*dt, so the solver executes steps/tau_scale of them and a wall time
    divided by the NOMINAL count overstates the per-step cost by exactly 1/tau_scale. Measured:
    the same point reported 0.139, 0.542 and 1.071 ms per step at nominal steps of 50, 200 and
    400 us purely through that factor. Cost is therefore measured with the controller OFF, where
    the executed count equals the nominal one, and the ceiling is measured in a separate run with
    the controller ON, where the wall time is discarded.
    """
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'adv.lm')
        build(path, steps, 1, dt)
        solver = lm.DriftMpdRdmeSolver()
        configure(solver, adaptive)
        t0 = time.time()
        lm.runSolver(path, 1, solver, [A.gpu], 0)
        wall = time.time() - t0
        tau_scale = float(solver.getTauScale()) if adaptive else 1.0
        sum_max = float(solver.getLastSumMax()) if adaptive else float('nan')
        with h5py.File(path, 'r') as f:
            g = f['/Simulations/0000001/Lattice']
            last = sorted(g.keys())[-1]
            lattice = g[last][...]
        # occupancy per species per voxel: the byte lattice stores species ids 1..nspec in the
        # trailing particle axis of each site.
        occ = np.stack([(lattice == i + 1).sum(axis=-1).ravel() for i in range(len(names))])
    return wall, tau_scale, sum_max, occ


def propensities(occ):
    """Total reaction propensity per voxel, 1/s, from the kernel-convention rate table."""
    idx = {n: i for i, n in enumerate(names)}
    a = np.zeros(occ.shape[1], float)
    per_channel = []
    for kind, react, prod, rate in S.rates(A.scheme, 'kernel'):
        if kind == 'fusion':
            ch = rate * occ[idx[react[0]]].astype(float) * occ[idx[react[1]]].astype(float)
        else:
            ch = rate * occ[idx[react[0]]].astype(float)
        per_channel.append(float(ch.max()))
        a += ch
    return a, per_channel


# COST: controller off, so the executed step count is the nominal one.
w1, _, _, _ = run_once(A.steps1, A.dt, False)
w2, _, _, occ = run_once(A.steps2, A.dt, False)
per_step = (w2 - w1) / float(A.steps2 - A.steps1)

# CEILING: controller on at a larger nominal step so it engages and its feedback averages over
# steps. tau_samp = dt_ceiling * tau_scale at convergence. The wall time of this run is discarded.
if A.no_adaptive:
    ts2, sm2, tau_samp = 1.0, float('nan'), float('nan')
else:
    _, ts2, sm2, occ_c = run_once(max(2000, A.steps1), A.dt_ceiling, True)
    tau_samp = A.dt_ceiling * ts2
    # ★Every occupancy statistic and the reaction propensity are taken from THIS run, not from the
    # cost runs. The controller holds this one at the declared accuracy budget, whereas a cost run
    # at a fixed nominal step can sit past it and then reports a broadened occupancy field: the
    # same point gave occ_max 11 at the budget and 19 past it, and a_max 1200 against 3160 /s.
    occ = occ_c

a_v, per_channel = propensities(occ)
tot = occ.sum(axis=0)
hist = np.bincount(tot, minlength=1)

meta = dict(engine='lat_gpu', scheme=A.scheme, eta=A.eta, dims=list(S.DIMS), M=M,
            species=names, sigma=[float(x) for x in sig], counts0=counts0,
            n0=int(sum(counts0.values())), dt_s=A.dt, eps=A.eps,
            tau_scale=ts2, sum_max=sm2, dt_ceiling=A.dt_ceiling, tau_samp_s=tau_samp,
            steps1=A.steps1, steps2=A.steps2, wall1_s=w1, wall2_s=w2, per_step_s=per_step,
            seed=A.seed, exclusion=not A.no_exclusion, field=not A.no_field,
            adaptive=not A.no_adaptive,
            counts_final=[int(x) for x in occ.sum(axis=1)],
            occ_mean=float(tot.mean()), occ_var=float(tot.var()),
            occ_max=int(tot.max()), occ_hist=[int(x) for x in hist],
            a_max_per_s=float(a_v.max()), a_mean_per_s=float(a_v.mean()),
            a_channel_max=per_channel,
            hardware='NVIDIA B200')
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
json.dump(meta, open(A.out, 'w'))
print('ADV_LAT %s eta=%.3f n0=%d per_step=%.5gs tau_samp=%.3g us (tau_scale %.4g, sum_max %.4g) '
      'a_max=%.4g /s occ %.2f+-%.2f max %d'
      % (A.scheme, A.eta, meta['n0'], per_step, tau_samp * 1e6, ts2, sm2, meta['a_max_per_s'],
         meta['occ_mean'], math.sqrt(meta['occ_var']), meta['occ_max']))
