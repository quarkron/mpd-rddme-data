"""ReaDDy arm of the polymer-crowding demixing scene, for the per-step cost.

WHAT IS MATCHED, AND WHY IT IS EXPENSIVE. The scene is built by adv_demix_scene, the module the
transport arm uses too, so the two price and measure one system: a 3 nm bath at volume fraction
0.05, 20 nm crowders at 0.20 and 60 000 polymer subunits of 10 nm in a 640 nm cube, the crowder
pair repulsion kept (it sets a 20 nm cutoff for the whole system, a real cost of the matched
physics), chains bonded at the lattice chain's sqrt(2) h, growing and shrinking through topology
reactions so the chemistry the lattice prices is priced here too.

WHICH STATE IS TIMED. A per-step cost is a property of the configuration it is measured in, and the
two engines must be timed in the same one. With --state-from the chains are the lattice run's own
final backbones, bead for bead, with the unbound remainder as free monomer, and the crowders its last
snapshot; this is the state the paper's demixing figure reports at 10 s. Without it the chains are
straight strings of --mean-len holding every monomer, which is what the Sep 22 and Sep 25 cost runs
did (and, before 2026-09-28, at a 10 nm bond rather than sqrt(2) h, in a private copy of the scene).

TIMING. Two-point differencing inside one process on one simulation object, which cancels setup and
the lazy neighbour-list build. A matched trajectory is not attempted: 10 s at this engine's
stability step is of order 10^8 steps.

Pin externally:  taskset -c <core> python adv_readdy_demix.py ...
"""
import argparse, json, os, sys, time

import numpy as np
import readdy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adv_demix_scene as SC

ap = SC.add_args(argparse.ArgumentParser())
ap.add_argument('--steps1', type=int, default=2)
ap.add_argument('--steps2', type=int, default=12)
ap.add_argument('--out', required=True)
A = ap.parse_args()

r = SC.resolve(A)
system = SC.build_system(A, r)
sim = SC.make_simulation(A, system)
n_seeded = SC.seed(A, r, sim, np.random.default_rng(A.seed))
print(SC.describe(A, r, n_seeded), flush=True)

t0 = time.perf_counter(); sim.run(A.steps1, r['dt'], show_summary=False)
t1 = time.perf_counter(); sim.run(A.steps2 - A.steps1, r['dt'], show_summary=False)
t2 = time.perf_counter()
w1, w2 = t1 - t0, t2 - t1
per_step = w2 / float(A.steps2 - A.steps1)

st = r['state']
meta = dict(engine='readdy', scheme='demix', arm=A.arm, kernel=('SingleCPU' if A.arm == 'sc' else 'CPU'),
            threads=(A.threads if A.arm == 'cpu' else 1), box=r['box'], nx=A.nx,
            n0=r['n0'], n_bath=r['n_bath'], n_crowd=r['n_crowd'], n_monomer=r['n_mono'],
            n_chain=r['n_chain'], n_bound=n_seeded, bond_nm=SC.BOND_NM,
            state_from=(st['path'] if st else None), state_t=(st['t'] if st else None),
            crowders_from_state=bool(st and st['crowders'] is not None),
            mean_len=(n_seeded / r['n_chain'] if r['n_chain'] else 0), sigma=r['sig'], D=r['D'],
            phi_crowd=A.phi_crowd, bath_eta=(0 if A.no_bath else A.bath_eta),
            crowder_pair=not A.no_crowder_pair, chains=('none' if A.no_chains else
                                                        'static' if A.static_chains else 'reactive'),
            k_poly=A.k_poly, k_depoly=A.k_depoly,
            dt=r['dt'], tau_stability=r['tau_stab'], steps1=A.steps1, steps2=A.steps2, seed=A.seed,
            wall1_s=w1, wall2_s=w2, per_step_s=per_step, readdy=readdy.__version__)
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
json.dump(meta, open(A.out, 'w'))
print('ADV_DEMIX_READDY %s N=%d per_step=%.5gs (windows %.1f/%.1f s)'
      % (A.arm, r['n0'], per_step, w1, w2), flush=True)
