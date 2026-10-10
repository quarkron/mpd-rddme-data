"""Ternary-mixture arm of the advantage metric: the lattice side. Runs in the lm2.6 container.

The main-text ternary scene of the manuscript, but sized to a box the particle engine can also
time: 32 x 8 x 8 subvolumes = 320 x 80 x 80 nm, holding 34225 A (2 nm), 4278 B (4 nm) at
eta = 0.07 each, and 4 oversize C bodies (20 nm, twice the subvolume edge) at one per (80 nm)^3,
which is the same number density as the published 32^3 scene and the same as the ReaDDy arm.

Differences from testing/engine/ternary_relax.py, and why:
  * uniform start, not the striped one. The stripe exists to make the RELAXATION comparable to
    the reference; a ceiling is a stationary property and the transient would bias it.
  * no recording hook. The hook costs wall time and would land in the per-step cost.
  * cost measured with the adaptive controller OFF over a fixed step count, ceiling measured with
    it ON in a separate run. lm.runSolver advances to a simulated TIME, so with the controller
    engaged the executed step count is steps/tau_scale and a cost divided by the nominal count is
    wrong by exactly that factor. Same trap, same fix, as adv_lattice.py.
"""
import argparse, json, math, os, sys, tempfile, time
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ENG = os.environ.get('LM_ENGINE_DIR', '/lm/testing/engine')
sys.path.insert(0, os.environ.get('LM_TESTING_DIR', '/lm/testing'))
sys.path.insert(0, ENG)
from lm_build import resolve_lm_build, report_imported_lm
resolve_lm_build()
import lm
report_imported_lm()
from pyLM import RDME
from pyLM.units import nm, micron, microsecond
import ribosome_blob as rb
import bath_species as bsp
import bath_scene as scene
import adv_schemes as S

NM = 10.0
V = NM ** 3
TAU_S = 50e-6
D_A_UM2 = 0.1
D1_UM2 = 0.9
SIGMA_FILL = 5.0
LAM_HC = 1.0
BATH_ID, RIBO_ID, D_MODEL_KIRKWOOD = 1, 2, 3
DIR_NONE, ROLE_BODY, FLAG_ACTIVE, FLAG_INERT = 255, 0, 1, 0x10

ap = argparse.ArgumentParser()
ap.add_argument('--dims', default='32,8,8')
ap.add_argument('--sigma-a', type=float, default=2.0)
ap.add_argument('--sigma-b', type=float, default=4.0)
ap.add_argument('--sigma-c', type=float, default=20.0)
ap.add_argument('--eta-a', type=float, default=0.07)
ap.add_argument('--eta-b', type=float, default=0.07)
ap.add_argument('--steps1', type=int, default=300)
ap.add_argument('--steps2', type=int, default=1500)
ap.add_argument('--dt', type=float, default=TAU_S)
ap.add_argument('--dt-ceiling', type=float, default=2e-4)
ap.add_argument('--eps', type=float, default=0.2)
ap.add_argument('--hop-sensor', dest='hop_sensor', type=int, default=1,
                help='1 = the rigid body hop reports into the ceiling sensor (default); 0 = the '
                     'subvolume hop alone, which is what the engine measured before it was armed')
ap.add_argument('--point', action='store_true', help='point-particle limit: no exclusion anywhere')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--out', required=True)
A = ap.parse_args()

NX, NY, NZ = (int(v) for v in A.dims.split(','))
XYZ = NX * NY * NZ
sig = {'A': A.sigma_a, 'B': A.sigma_b, 'C': A.sigma_c}
D_UM2 = {'A': D_A_UM2, 'B': D_A_UM2 * sig['A'] / sig['B'], 'C': D_A_UM2 * sig['A'] / sig['C']}
NA = int(round(A.eta_a * XYZ * V / (np.pi / 6 * sig['A'] ** 3)))
Bspec = bsp.parse(['%g:%g:%g' % (sig['B'], A.eta_b, D_UM2['B'])], D_UM2['C'], sig['C'])
excl = not A.point
# one body per (80 nm)^3, placed on a regular grid, which is the published number density
PER = 8
cx = [PER * k + PER // 2 for k in range(max(1, NX // PER))]
cy = [PER * k + PER // 2 for k in range(max(1, NY // PER))]
cz = [PER * k + PER // 2 for k in range(max(1, NZ // PER))]
centers = [(x, y, z) for x in cx for y in cy for z in cz]
vox = lambda x, y, z: z * (NX * NY) + y * NX + x


def configure(sol, adaptive):
    sol.setPotentialBasis(np.zeros(XYZ, np.float32), np.zeros(XYZ, np.float32))
    sol.setStructureHandler(1, SIGMA_FILL, V)
    sol.configureStructureLattice(NX, NY, NZ)
    diam = bsp.diameters(Bspec, sig['A'], SIGMA_FILL)
    sol.setSpeciesHardSphereDiameters(diam, V)
    for t in range(1, len(diam) + 1):
        sol.setSpeciesCoupling(t - 1, 0.0, 0.0)
    if excl:
        sol.setHardCoreExclusion(LAM_HC,
                                 lm.HCForm_WHITE_BEAR if hasattr(lm, 'HCForm_WHITE_BEAR') else 2)
    l1 = np.zeros(XYZ, np.int32); l2 = np.full(XYZ, DIR_NONE, np.int32); l3 = np.zeros(XYZ, np.int32)
    coms, flags, nsub = [], [], []
    for i, (x, y, z) in enumerate(centers):
        v = vox(x, y, z); l1[v] = i + 1; l3[v] = (ROLE_BODY << 4) | 1
        coms += [float(x), float(y), float(z)]
        nsub.append(1)
        flags.append(FLAG_ACTIVE | FLAG_INERT | (RIBO_ID << 8))
    sol.setStructureOverlay(l1, l2, l3)
    sol.setStructureRegistry(len(centers), np.array(nsub, np.int32),
                             np.array(coms, np.float64), np.array(flags, np.int32))
    for i, (x, y, z) in enumerate(centers):
        sol.setStructureTips(i, vox(x, y, z), vox(x, y, z), -1)
    sol.setStructureSizeDiffusion(D_MODEL_KIRKWOOD, D1_UM2 * 1e6, TAU_S, NM)
    sol.setStructureHopSpeciesScale(RIBO_ID, D_UM2['C'] / D1_UM2)
    if excl:
        sol.setStructureSmearStencil(rb.fmt_stencil(sig['C'], NM).reshape(-1).astype(np.float32))
        sol.setStructureSmearDnaBase(np.zeros(4 * XYZ, np.float64))
        sol.armStructureSmearRefresh(RIBO_ID)
        sol.setStructureSmearIncremental(512)
        sol.setStructureReciprocalVex(True, BATH_ID)
        sol.setStructureFreeVolume(1)
    else:
        sol.setStructureCollisionalFriction(0)
    sol.setHopIntegration(1)
    # The body in this scene hops by a sub-subvolume step, so its ceiling is NOT the bath's. With
    # the sensor off the controller reports the bath alone, which is what the first version of the
    # main-text ternary row quoted.
    try:
        sol.setStructureHopSensor(int(A.hop_sensor))
    except AttributeError:
        print('[tern] build predates setStructureHopSensor: ceiling is the subvolume hop alone')
    if adaptive:
        sol.setAdaptiveTau(1, float(A.eps), 0.05)
    sol.validateStability()


def run_once(steps, dt, adaptive):
    box = micron(NX * NM * 1e-3, NY * NM * 1e-3, NZ * NM * 1e-3)
    sim = RDME.RDMESimulation(dimensions=box, spacing=nm(NM))
    sim.defineSpecies(['A'])
    sim.modifyRegion('default').setDefaultDiffusionRate(D_A_UM2 * 1e-12)
    bsp.define_in_sim(sim, Bspec, NX, NY, NZ, NM)
    sim.addParticles(species='A', region='default', count=NA)
    # Hooks OFF: a recording hook costs wall time and would land in the per-step cost. One
    # lattice dump at the end, which is where the occupancy statistics come from. A fractional
    # interval truncates to 0 and means EVERY step, so it is passed as an integer step count.
    scene.finish_sim(sim, A.seed, dt, steps * dt, scene.NEVER_STEPS,
                     lattice_write_steps_=max(1, int(steps)))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'tern.lm')
        sim.save(path)
        sol = lm.DriftMpdRdmeSolver()
        configure(sol, adaptive)
        t0 = time.time()
        lm.runSolver(path, 1, sol, [A.gpu], 0)
        wall = time.time() - t0
        ts = float(sol.getTauScale()) if adaptive else 1.0
        sm = float(sol.getLastSumMax()) if adaptive else float('nan')
        err = np.zeros(4, np.int32); sol.getStructureError(err)
        with h5py.File(path, 'r') as f:
            g = f['/Simulations/0000001/Lattice']
            keys = sorted(g.keys())
            lat = g[keys[-1]][...] if keys else None
    occ = None
    if lat is not None:
        occ = np.stack([(lat == t).sum(axis=-1).ravel() for t in (1, 3)])
    return wall, ts, sm, int(err[0]), occ


w1, _, _, e1, _ = run_once(A.steps1, A.dt, False)
w2, _, _, e2, occ = run_once(A.steps2, A.dt, False)
per_step = (w2 - w1) / float(A.steps2 - A.steps1)
_, ts, sm, e3, occ_c = run_once(max(1000, A.steps1), A.dt_ceiling, True)
tau_samp = A.dt_ceiling * ts
if occ_c is not None:
    occ = occ_c

tot = occ.sum(axis=0) if occ is not None else np.zeros(1, int)
meta = dict(engine='lat_gpu', scheme='ternary', dims=[NX, NY, NZ], M=XYZ,
            sigma=sig, eta_a=A.eta_a, eta_b=A.eta_b, NA=NA, NB=int(Bspec[0].count),
            n_c=len(centers), n0=NA + int(Bspec[0].count), point=bool(A.point),
            dt_s=A.dt, dt_ceiling=A.dt_ceiling, eps=A.eps, hop_sensor=int(A.hop_sensor),
            tau_scale=ts, sum_max=sm, tau_samp_s=tau_samp,
            steps1=A.steps1, steps2=A.steps2, wall1_s=w1, wall2_s=w2, per_step_s=per_step,
            struct_err=[e1, e2, e3], seed=A.seed,
            occ_mean=float(tot.mean()), occ_var=float(tot.var()), occ_max=int(tot.max()),
            a_max_per_s=0.0, a_mean_per_s=0.0, hardware='NVIDIA B200')
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
json.dump(meta, open(A.out, 'w'))
print('ADV_TERN dims=%dx%dx%d NA=%d NB=%d C=%d per_step=%.5gs tau_samp=%.3g us '
      '(tau_scale %.4g sum_max %.4g) occ %.2f max %d struct_err=%s'
      % (NX, NY, NZ, NA, Bspec[0].count, len(centers), per_step, tau_samp * 1e6, ts, sm,
         meta['occ_mean'], meta['occ_max'], meta['struct_err']))
