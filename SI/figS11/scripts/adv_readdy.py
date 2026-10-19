"""ReaDDy timing arm of the advantage-metric campaign.

Three arms, because the cost ratio is only honest if each engine is allowed its best kernel:

  sc_db    SingleCPU + DetailedBalance handler, one pinned core. MATCHED PHYSICS: this is the
           construction the lattice reaction acceptance mirrors, and it is the arm the accuracy
           chapters validated against. The handler is implemented for SingleCPU only, which the
           engine states outright, so this arm cannot use more than one core.
  cpu_g    CPU kernel + Gillespie handler, one pinned core. BEST SINGLE-CORE KERNEL, different
           reaction physics: no detailed-balance correction, so exclusion and chemistry are not
           coupled the way the lattice couples them. Labelled as such wherever it is reported.
  cpu_gN   the same on a full socket, for the socket-to-socket comparison.

Timing is the two-point difference wall(steps2) - wall(steps1) taken inside one process on one
simulation object, which cancels import, setup and the lazy neighbour-list build. The second
segment continues the first, so both sample the running state rather than the artificial start.

Pin externally:  taskset -c <core> python adv_readdy.py ...
"""
import argparse, json, os, sys, time
import numpy as np
import readdy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'readdy_ref', 'sinpot'))
import adv_schemes as S

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', required=True, choices=sorted(S.SCHEMES))
ap.add_argument('--eta', type=float, required=True)
ap.add_argument('--arm', default='sc_db', choices=('sc_db', 'cpu_g'))
ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--dt', type=float, default=None, help='default: the Euler stability bound / 2.5')
ap.add_argument('--steps1', type=int, default=100)
ap.add_argument('--steps2', type=int, default=500)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--no-field', action='store_true')
ap.add_argument('--out', required=True)
A = ap.parse_args()

spec = S.SCHEMES[A.scheme]
names = spec['species']
sig = spec['sigma']
D = S.diffusion(A.scheme)
box = [S.DIMS[0] * S.H_NM, S.DIMS[1] * S.H_NM, S.DIMS[2] * S.H_NM]

# The engine's own stability limit for the stiffest contact it integrates, and the step actually
# used. The published arms run a factor of 2.5 inside the limit; keeping that ratio here means the
# ceiling ratio is reported against the LIMIT while the cost is measured at a step the engine is
# known to be stable at. Both numbers are recorded.
tau_stab = 2.0 / (S.D_NM2S * S.K_SOFT)
dt = A.dt if A.dt is not None else tau_stab / 2.5

system = readdy.ReactionDiffusionSystem(box, unit_system=None)
system.kbt = 1.0
for n in names:
    system.add_species(n, D[n])

# ★The body-body pair is EXCLUDED for the ternary scene, deliberately and to match the lattice.
# The lattice arm carries its oversize spheres with FLAG_INERT, so they do not interact with each
# other; adding a C-C repulsion here would be different physics AND would set the neighbour cutoff
# to the 20 nm body diameter for the whole system, which the published reference measures at about
# forty times slower. Keeping it made a 50-step timing run exceed two minutes without finishing.
inert_self = {'C'} if A.scheme == 'ternary' else set()
done = set()
for a in names:
    for b in names:
        key = tuple(sorted((a, b)))
        if key in done:
            continue
        done.add(key)
        if a == b and a in inert_self:
            continue
        system.potentials.add_harmonic_repulsion(
            a, b, S.K_SOFT, S.bh_contact(0.5 * (sig[a] + sig[b])))

lam = S.doi_lambda(A.scheme)
for kind, react, prod, rate in S.rates(A.scheme, 'pylm'):
    if kind == 'fusion':
        system.reactions.add_fusion('f_%s%s' % react, react[0], react[1], prod[0], lam, S.R_DOI)
    elif kind == 'fission':
        system.reactions.add_fission('r_%s' % react[0], react[0], prod[0], prod[1],
                                     rate, S.R_DOI)
    else:
        system.reactions.add_conversion('c_%s%s' % (react[0], prod[0]), react[0], prod[0], rate)

if not A.no_field:
    import readdy_sinpot
    for n in names:
        system.potentials.add_custom_external(n, readdy_sinpot.Sinusoid, S.GAMMA,
                                              S.WAVELENGTH, 0.0)

kernel = 'SingleCPU' if A.arm == 'sc_db' else 'CPU'
sim = system.simulation(kernel=kernel)
if A.arm == 'sc_db':
    if spec['reactions']:
        sim.reaction_handler = 'DetailedBalance'
else:
    sim.reaction_handler = 'Gillespie'
    try:
        sim.kernel_configuration.n_threads = int(A.threads)
    except Exception as exc:
        print('WARNING: could not set n_threads: %s' % exc)
sim.output_file = ''

counts = S.counts_for_eta(A.scheme, A.eta)
rng = np.random.default_rng(A.seed)
half = np.array(box) / 2.0
n_total = 0
for n, c in counts.items():
    if c:
        sim.add_particles(n, rng.uniform(-half, half, size=(int(c), 3)))
        n_total += int(c)

t0 = time.perf_counter(); sim.run(A.steps1, dt, show_summary=False)
t1 = time.perf_counter(); sim.run(A.steps2 - A.steps1, dt, show_summary=False)
t2 = time.perf_counter()
w1, w2 = t1 - t0, t2 - t1
per_step = w2 / float(A.steps2 - A.steps1)

meta = dict(engine='readdy', arm=A.arm, kernel=kernel, threads=(A.threads if A.arm != 'sc_db' else 1),
            handler=('DetailedBalance' if A.arm == 'sc_db' else 'Gillespie'),
            scheme=A.scheme, eta=A.eta, box=box, species=names,
            sigma={k: float(v) for k, v in sig.items()}, n0=n_total, counts0=counts,
            dt=dt, tau_stability=tau_stab, k_soft=S.K_SOFT,
            steps1=A.steps1, steps2=A.steps2, seed=A.seed, field=not A.no_field,
            inert_self=sorted(inert_self),
            wall1_s=w1, wall2_s=w2, per_step_s=per_step, readdy=readdy.__version__)
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
json.dump(meta, open(A.out, 'w'))
print('ADV_READDY %s %s eta=%.3f n0=%d threads=%d per_step=%.5gs (windows %.2f/%.2f s)'
      % (A.scheme, A.arm, A.eta, n_total, meta['threads'], per_step, w1, w2))
