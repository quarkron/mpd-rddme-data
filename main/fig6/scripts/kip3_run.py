"""Run one arm of the Kip3 traffic problem on the MPD-RDDME lattice and reduce it to an npz.

    python kip3_run.py --mode traffic --arm excluded --lanes switching --leduc fig1c \
                       --tsim 600 --out out/fig1c_exc_sw.npz

MODES
    traffic   Leduc's experiment: empty track at t = 0, attachment everywhere, detachment on the
              lattice at k_off and at the plus end at k_end. Rates from --leduc or the explicit
              --kon-N/--rho-s/--kend-bar triple.
    passing   gate: two motors on adjacent sites of one lane, no kinetics. With exclusion the
              rear one never passes; without it the two share a voxel within seconds.
    attach    gate: lane 0 filled to one motor per site, attachment only (k_off = k_end = 0).
              Lane 0 must keep exactly N motors (no attachment into occupied sites) while the
              empty lanes fill as 1 - exp(-k_on t), which measures the realised k_on.
    dilute    gate: ten tracers per lane 100 sites apart, no kinetics. Speed must be v0.

ARMS      excluded  hard-sphere exclusion on (White Bear, sigma from kip3_model)
          point     exclusion off: the control, and the argument
LANES     isolated  no lateral hops: sixteen copies of Leduc's one-dimensional model
          switching lateral hops at Kip3's measured sidestep rate

GEOMETRY  Solver axes (x, y, z) = (1056, 8, 8); the track runs along x from X0 over --sites
          voxels. Body 3x3 at y,z in [3,5]; shell ring y,z in [2,6] minus the body = 16 lanes,
          typed A/B alternately around the ring so every lateral hop crosses a type boundary and
          is rated by its own transition rate. The last ring is tipA/tipB for the end kinetics.
          Periodic boundaries: 28 inert voxels at each end of x and one inert layer in y and z
          keep the wrap harmless (no transition enters cyt or body).

INDEX-ORDER TRAPS (inherited from synapse_run.py, still true)
    jLM reverses axes: dimensions are given as [nz, ny, nx], masks as (z, y, x) via to_jlm, and
    placeNumber takes (z, y, x). Lattice particle types are 1-based; setSpeciesCoupling and
    setSpeciesHardSphereDiameters are 0-based (idx - 1). Species are created M, T, C in that
    order so that a two-entry diameter vector configures M and T and leaves C unconfigured,
    which is what makes C invisible to the functional (a configured sigma = 0 would still enter
    xi_0 and charge -ln(1 - eta) per copy).
"""
import argparse, json, math, os, sys, tempfile, time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M

import lm
from jLM.RDME import Sim as RDMESim
from jLM.RegionBuilder import RegionBuilder
from jLM.units import nm

NX, NY, NZ = 1056, 8, 8
X0 = 28                    # first track site
LO, HI = 2, 6              # ring extent in y and z; body is [3, 5]


def to_jlm(a):
    """(x, y, z) in solver order -> the (z, y, x) order jLM's masks and dimensions use."""
    return np.ascontiguousarray(a.transpose(2, 1, 0))


def ring_order():
    """The 16 shell voxels (y, z) in cyclic order around the body; lane l is ring[l]."""
    ring = [(y, LO) for y in range(LO, HI + 1)]
    ring += [(HI, z) for z in range(LO + 1, HI + 1)]
    ring += [(y, HI) for y in range(HI - 1, LO - 1, -1)]
    ring += [(LO, z) for z in range(HI - 1, LO, -1)]
    assert len(ring) == 16 and len(set(ring)) == 16
    return ring


def masks(L):
    shape = (NX, NY, NZ)
    body = np.zeros(shape, bool); body[X0:X0 + L, 3:6, 3:6] = True
    lane = [np.zeros(shape, bool), np.zeros(shape, bool)]
    tip = [np.zeros(shape, bool), np.zeros(shape, bool)]
    ring = ring_order()
    for l, (y, z) in enumerate(ring):
        lane[l % 2][X0:X0 + L - 1, y, z] = True
        tip[l % 2][X0 + L - 1, y, z] = True
    return body, lane[0], lane[1], tip[0], tip[1], ring


def potential_basis(L, shell):
    """psi0 = alpha (L-1-s) on the track, zero at the plus end (see kip3_model, fact 1); zero
    elsewhere. Flattened in the kernel's (z, y, x) order."""
    psi = np.zeros((NX, NY, NZ), np.float64)
    s = np.arange(NX) - X0
    tilt = M.ALPHA_KT * (L - 1 - s)
    psi[shell] = np.broadcast_to(tilt[:, None, None], psi.shape)[shell]
    flat = np.ascontiguousarray(psi.transpose(2, 1, 0).ravel(order='C').astype(np.float32))
    return flat, np.zeros(NX * NY * NZ, np.float32)


def kinetics(A):
    """(k_on, k_off, k_end) in s^-1 for the mode, or None when the mode has no kinetics."""
    if A.mode in ('passing', 'dilute'):
        return None
    if A.leduc != 'none':
        kon_N, rho_s, kend = M.LEDUC[A.leduc]
    else:
        kon_N, rho_s, kend = A.kon_N, A.rho_s, A.kend_bar
    if A.kon_N is not None: kon_N = A.kon_N
    if A.rho_s is not None: rho_s = A.rho_s
    if A.kend_bar is not None: kend = A.kend_bar
    if None in (kon_N, rho_s, kend):
        raise SystemExit('traffic mode needs kon_N, rho_s and kend_bar (a preset or all three)')
    k_on, k_off, k_end = M.leduc_rates(kon_N, rho_s, kend, A.sites)
    if A.mode == 'attach':
        k_off = k_end = 0.0
    return k_on, k_off, k_end


def build(path, A, rates):
    L = A.sites
    dt = A.q_stab * (M.H_NM * 1e-9) ** 2 / M.D_AX
    sim = RDMESim(name='kip3', filename=path, dimensions=[NZ, NY, NX],
                  latticeSpacing=nm(M.H_NM), regionName='cyt', latticeType='Byte', dt=dt)
    m_body, m_A, m_B, m_tA, m_tB, ring = masks(L)
    shell = m_A | m_B | m_tA | m_tB
    cyt = sim.region('cyt'); body = sim.region('body')
    laneA = sim.region('laneA'); laneB = sim.region('laneB')
    tipA = sim.region('tipA'); tipB = sim.region('tipB')
    RegionBuilder(sim).compose((cyt, np.ones((NZ, NY, NX), bool)), (body, to_jlm(m_body)),
                               (laneA, to_jlm(m_A)), (laneB, to_jlm(m_B)),
                               (tipA, to_jlm(m_tA)), (tipB, to_jlm(m_tB)))
    lanes, tips = [laneA, laneB], [tipA, tipB]

    # Species order matters: M = 1, T = 2, C = 3 (see the index-order traps above).
    Msp = sim.species('M'); Tsp = sim.species('T'); Csp = sim.species('C')
    species = {'M': Msp, 'T': Tsp, 'C': Csp}

    # Every transition is closed, then only the track is opened. Axial hops stay within a lane
    # type; lateral hops cross A<->B and carry their own rate. C never moves.
    sim.transitionRate(None, None, None, sim.diffusionZero)
    d_ax = sim.diffusionConst('D_ax', M.D_AX)
    axial = [(laneA, laneA), (laneB, laneB), (laneA, tipA), (tipA, laneA),
             (laneB, tipB), (tipB, laneB)]
    lateral = [(laneA, laneB), (laneB, laneA), (tipA, tipB), (tipB, tipA)]
    for sp in (Msp, Tsp):
        for a, b in axial:
            sim.transitionRate(sp, a, b, d_ax)
    if A.lanes == 'switching':
        d_lat = sim.diffusionConst('D_lat', M.D_LAT)
        for sp in (Msp, Tsp):
            for a, b in lateral:
                sim.transitionRate(sp, a, b, d_lat)

    # Kinetics. The reservoir species is what keeps the tilt out of the attachment acceptance;
    # detachment is a plain first-order decay, accepted with probability one under the plus-end
    # gauge. Rate names differ per reaction: jLM keys a reaction on reactants, products AND rate
    # name, so the two M -> 0 channels stay distinct.
    n_rx = 0
    k_on_nom = None
    if rates is not None:
        k_on, k_off, k_end = rates
        k_on_nom = k_on * (math.exp(M.F_INS) if A.arm == 'excluded' else 1.0)
        # The tracer channel is carved out of k_on, not added to it: the total attachment rate
        # per site is k_on and a fraction f_tracer of the arrivals are tracers.
        k_att = k_on_nom / A.n_res
        allsh = lanes + tips
        sim.reaction([Csp], [Msp], sim.rateConst('k_att_M', k_att * (1.0 - A.f_tracer), 1),
                     regions=allsh); n_rx += 4
        if A.f_tracer > 0:
            sim.reaction([Csp], [Tsp], sim.rateConst('k_att_T', k_att * A.f_tracer, 1),
                         regions=allsh); n_rx += 4
        if k_off > 0:
            for sp, nm_ in ((Msp, 'M'), (Tsp, 'T')):
                sim.reaction([sp], [], sim.rateConst('k_off_' + nm_, k_off, 1), regions=lanes)
                n_rx += 2
        if k_end > 0:
            for sp, nm_ in ((Msp, 'M'), (Tsp, 'T')):
                sim.reaction([sp], [], sim.rateConst('k_end_' + nm_, k_end, 1), regions=tips)
                n_rx += 2
        if len(sim._reactionLocations) != n_rx:
            raise SystemExit('reactions were not assigned to every region: %d of %d'
                             % (len(sim._reactionLocations), n_rx))

    # Seeding. The reservoir fills every shell voxel; the gates add their probes.
    seeded = {}
    if rates is not None:
        for x, y, z in np.argwhere(shell):
            sim.placeNumber(Csp, int(z), int(y), int(x), A.n_res)
        seeded['C'] = int(shell.sum()) * A.n_res
    y0, z0 = ring[0]
    if A.mode == 'passing':
        for s in (100, 101):
            sim.placeNumber(Msp, z0, y0, X0 + s, 1)
        seeded['M'] = 2
    elif A.mode == 'attach' and not A.no_full_lane:
        # Without exclusion and with k_end = 0 this lane collapses into the tip voxel and
        # overflows it; the point arm runs the gate with --no-full-lane (k_on check only).
        for s in range(L):
            sim.placeNumber(Msp, z0, y0, X0 + s, 1)
        seeded['M'] = L
    elif A.mode == 'dilute':
        n = 0
        spacing = max(1, (L - 100) // A.tracers_per_lane)
        for (y, z) in ring:
            for s in range(50, 50 + spacing * A.tracers_per_lane, spacing):
                sim.placeNumber(Tsp, z, y, X0 + s, 1); n += 1
        seeded['T'] = n

    steps = int(round(A.tsim / dt))
    per_frame = max(1, int(round(A.frame_dt / dt)))
    sim.timestep = dt
    sim.simulationTime = A.tsim
    sim.latticeWriteInterval = int(per_frame)      # STEP counts, not times
    sim.speciesWriteInterval = int(per_frame)
    return sim, species, dt, steps, per_frame, ring, shell, seeded, k_on_nom


def configure(solver, A, species, psi0, psi1):
    solver.setPotentialBasis(psi0, psi1)
    for sp in species.values():                      # M, T and C all ride the tilt (fact 1)
        solver.setSpeciesCoupling(sp.idx - 1, 1.0, 0.0)
    if A.arm == 'excluded':
        diam = np.array([M.SIGMA_NM, M.SIGMA_NM], np.float64)   # M and T; C left unconfigured
        solver.setSpeciesHardSphereDiameters(diam, float(M.H_NM ** 3))
        solver.setHardCoreExclusion(1.0, 2)
    solver.setHopIntegration(0)      # LINEAR: p = q B(dU), exactly detailed-balanced and an
                                     # exact per-step rate; q B(-alpha) = 0.05 at q = 0.005
    solver.validateStability()


def reduce_frames(path, L, ring, species):
    """Per frame: motors (M+T) and tracers per lane per site, reservoir count per site summed
    over lanes, the largest motor occupancy of any voxel, and species totals."""
    ys = np.array([y for y, z in ring]); zs = np.array([z for y, z in ring])
    idx = [species[k].idx for k in ('M', 'T', 'C')]
    with h5py.File(path, 'r') as f:
        grp = f['/Simulations/0000001']
        lat = grp['Lattice']
        keys = sorted(lat.keys())
        times = grp['LatticeTimes'][...][:len(keys)]
        F = len(keys)
        dens = np.zeros((F, 16, L), np.uint8); trac = np.zeros((F, 16, L), np.uint8)
        cres = np.zeros((F, L), np.uint16); maxocc = np.zeros(F, np.int32)
        ntot = np.zeros((F, 3), np.int64)
        for fi, k in enumerate(keys):
            blk = lat[k][...]
            xax = [a for a, n in enumerate(blk.shape[:3]) if n == NX]
            if len(xax) != 1:
                raise SystemExit('cannot identify the x axis in lattice shape %r' % (blk.shape,))
            rest = [a for a in range(3) if a != xax[0]]
            # The two 8-axes cannot be told apart by shape. A y/z swap reflects the ring,
            # lane l -> -l mod 16, which preserves the A/B parity and every statistic.
            blk = blk.transpose((xax[0], rest[0], rest[1], 3))
            occ = [(blk == i).sum(-1) for i in idx]
            sl = slice(X0, X0 + L)
            mm = occ[0][sl][:, ys, zs]; tt = occ[1][sl][:, ys, zs]; cc = occ[2][sl][:, ys, zs]
            dens[fi] = (mm + tt).T; trac[fi] = tt.T; cres[fi] = cc.sum(1)
            maxocc[fi] = int((mm + tt).max())
            ntot[fi] = [int(o.sum()) for o in occ]
    return dict(times=times, dens=dens, trac=trac, cres=cres, maxocc=maxocc, ntot=ntot,
                ring=np.array(ring))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('traffic', 'passing', 'attach', 'dilute'), default='traffic')
    p.add_argument('--arm', choices=('excluded', 'point'), default='excluded')
    p.add_argument('--lanes', choices=('isolated', 'switching'), default='isolated')
    p.add_argument('--leduc', choices=tuple(M.LEDUC) + ('none',), default='none')
    p.add_argument('--kon-N', type=float, default=None, help="Leduc's kbar_on N")
    p.add_argument('--rho-s', type=float, default=None, help='Langmuir density')
    p.add_argument('--kend-bar', type=float, default=None, help='end dissociation per step time')
    p.add_argument('--tsim', type=float, default=600.0, help='physical seconds')
    p.add_argument('--frame-dt', type=float, default=2.0, help='seconds between frames')
    p.add_argument('--q-stab', type=float, default=0.005,
                   help='dt = q h^2 / D_ax; the forward hop probability per step is q B(-alpha)')
    p.add_argument('--sites', type=int, default=M.N_SITES)
    p.add_argument('--n-res', type=int, default=M.N_RES)
    p.add_argument('--f-tracer', type=float, default=M.F_TRACER)
    p.add_argument('--no-full-lane', action='store_true',
                   help='attach mode: do not seed lane 0 full (needed for the point arm)')
    p.add_argument('--tracers-per-lane', type=int, default=10,
                   help='dilute mode: tracers seeded per lane, evenly spaced from site 50')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--cuda-device', type=int, default=0, help='index INSIDE the container')
    p.add_argument('--keep-lm', default='')
    p.add_argument('--out', required=True)
    A = p.parse_args()
    if A.sites + 2 * X0 > NX:
        raise SystemExit('track of %d sites does not fit in %d with %d margins' % (A.sites, NX, X0))

    rates = kinetics(A)
    print(M.summary())
    print('mode=%s arm=%s lanes=%s leduc=%s  rates=%s' % (A.mode, A.arm, A.lanes, A.leduc,
          None if rates is None else 'k_on=%.4g k_off=%.4g k_end=%.4g s^-1' % rates))
    sys.stdout.flush()

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        path = A.keep_lm or os.path.join(td, 'kip3.lm')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        sim, species, dt, steps, per_frame, ring, shell, seeded, k_on_nom = build(path, A, rates)
        print('dt=%.4e s  steps=%d  frames every %d steps (%.2f s)  seeded %s  k_on_nom=%s'
              % (dt, steps, per_frame, per_frame * dt, seeded,
                 'n/a' if k_on_nom is None else '%.4g' % k_on_nom)); sys.stdout.flush()
        solver = lm.DriftMpdRdmeSolver()
        psi0, psi1 = potential_basis(A.sites, shell)
        configure(solver, A, species, psi0, psi1)
        sim.finalize()
        # jLM's own seed path calls lm.simulationFile, which this build does not export; the
        # wrapper is lm.SimulationFile. The solver reads the "seed" parameter from the file.
        sf = lm.SimulationFile(path)
        sf.setParameter('seed', str(A.seed))
        getattr(sf, 'close', lambda: None)()
        del sf
        t_run = time.time()
        sim.run(solver=solver, cudaDevices=[A.cuda_device])
        wall = time.time() - t_run
        print('run %.1f s wall  (%.1f us/step); word mark %d'
              % (wall, 1e6 * wall / max(steps, 1), int(solver.getLatticeMaxWord())))
        sys.stdout.flush()
        red = reduce_frames(path, A.sites, ring, species)

    meta = dict(mode=A.mode, arm=A.arm, lanes=A.lanes, leduc=A.leduc,
                rates=None if rates is None else dict(zip(('k_on', 'k_off', 'k_end'), rates)),
                k_on_nom=k_on_nom, kon_N=A.kon_N, rho_s=A.rho_s, kend_bar=A.kend_bar,
                tsim=A.tsim, dt=dt, steps=steps, frame_dt=A.frame_dt, q_stab=A.q_stab,
                sites=A.sites, n_res=A.n_res, f_tracer=A.f_tracer, seed=A.seed,
                h_nm=M.H_NM, alpha_kT=M.ALPHA_KT, v0_nm_s=M.V0_NM_S, D_ax=M.D_AX, D_lat=M.D_LAT,
                sigma_nm=M.SIGMA_NM, F_ins=M.F_INS, dims=[NX, NY, NZ], x0=X0,
                seeded=seeded, wall_s=wall)
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    np.savez_compressed(A.out, meta=json.dumps(meta), **red)
    print('wrote %s  (total %.1f s)  max motors in a voxel %d  final totals M,T,C = %s'
          % (A.out, time.time() - t0, int(red['maxocc'].max()), red['ntot'][-1].tolist()))


if __name__ == '__main__':
    main()
