"""One-dimensional TASEP with Langmuir kinetics, the reference for the isolated-lane arm.

    python tasep_lk_reference.py --leduc fig1c --lanes 16 --tsim 1500 --frame-dt 2 --out out/ref_fig1c.npz

This is the discrete-time process the lattice run realises lane by lane when lateral hops are
off: at every step of Delta_t each motor advances one site with probability p+ = q B(-alpha) if
the site ahead is empty, each empty site fills with probability k_on Delta_t (a fraction
f_tracer of the arrivals are tracers), each bound motor leaves with probability k_off Delta_t
on the lattice and k_end Delta_t on the last site. Updates are parallel (all sites at once), as
in the MPD passes. Leduc's own simulations used a random-sequential update with p+ = 1; the two
agree as p+ -> 0, and at p+ = 0.05 the parallel-update current is higher by a factor
(1 + p+ rho (1 - rho) + ...), under 3 % at rho = 1/2. The reference is therefore run at the
lattice's own p+ and Delta_t, and the lanes are independent copies.

Backward steps (probability p+ e^{-alpha} on the lattice) are omitted: 4.5e-5 of the forward
rate, below anything measured here. The output mirrors kip3_run.py (dens, trac, times, meta) so
kip3_analyze.py treats it as another arm.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M


def run(A):
    L, NL = A.sites, A.lanes
    dt = A.q_stab * (M.H_NM * 1e-9) ** 2 / M.D_AX
    p_fwd = A.q_stab * M.bernoulli(-M.ALPHA_KT)
    if A.leduc != 'none':
        kon_N, rho_s, kend = M.LEDUC[A.leduc]
    else:
        kon_N, rho_s, kend = A.kon_N, A.rho_s, A.kend_bar
    if A.kon_N is not None: kon_N = A.kon_N
    if A.rho_s is not None: rho_s = A.rho_s
    if A.kend_bar is not None: kend = A.kend_bar
    k_on, k_off, k_end = M.leduc_rates(kon_N, rho_s, kend, L)
    p_on, p_off, p_end = k_on * dt, k_off * dt, k_end * dt
    steps = int(round(A.tsim / dt))
    per_frame = max(1, int(round(A.frame_dt / dt)))
    F = steps // per_frame + 1
    rng = np.random.default_rng(A.seed)

    occ = np.zeros((NL, L), np.int8)          # 0 empty, 1 motor, 2 tracer
    dens = np.zeros((F, NL, L), np.uint8); trac = np.zeros((F, NL, L), np.uint8)
    times = np.zeros(F)
    fi = 0
    dens[0] = occ > 0; trac[0] = occ == 2
    t0 = time.time()
    for step in range(1, steps + 1):
        # forward hops: a motor at s moves to s+1 if s+1 is empty. Parallel update is
        # conflict-free for TASEP (a site is entered from one side only).
        mov = (occ[:, :-1] > 0) & (occ[:, 1:] == 0) & (rng.random((NL, L - 1)) < p_fwd)
        src = occ[:, :-1][mov]
        occ[:, 1:][mov] = src
        occ[:, :-1][mov] = 0
        # detachment on the lattice, end dissociation on the last site
        off = (occ > 0) & (rng.random((NL, L)) < p_off)
        off[:, -1] = (occ[:, -1] > 0) & (rng.random(NL) < p_end)
        occ[off] = 0
        # attachment into empty sites; a fraction f_tracer are tracers
        on = (occ == 0) & (rng.random((NL, L)) < p_on)
        kind = np.where(rng.random((NL, L)) < A.f_tracer, 2, 1).astype(np.int8)
        occ[on] = kind[on]
        if step % per_frame == 0:
            fi += 1
            dens[fi] = occ > 0; trac[fi] = occ == 2; times[fi] = step * dt
    dens, trac, times = dens[:fi + 1], trac[:fi + 1], times[:fi + 1]
    meta = dict(mode='traffic', arm='tasep_lk', lanes='isolated', leduc=A.leduc,
                rates=dict(k_on=k_on, k_off=k_off, k_end=k_end), kon_N=kon_N, rho_s=rho_s,
                kend_bar=kend, tsim=A.tsim, dt=dt, steps=steps, frame_dt=A.frame_dt,
                q_stab=A.q_stab, p_fwd=p_fwd, sites=L, n_lanes=NL, f_tracer=A.f_tracer,
                seed=A.seed, h_nm=M.H_NM, alpha_kT=M.ALPHA_KT, v0_nm_s=M.V0_NM_S,
                wall_s=time.time() - t0, update='parallel')
    return dens, trac, times, meta


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--leduc', choices=tuple(M.LEDUC) + ('none',), default='none')
    p.add_argument('--kon-N', type=float, default=None)
    p.add_argument('--rho-s', type=float, default=None)
    p.add_argument('--kend-bar', type=float, default=None)
    p.add_argument('--tsim', type=float, default=600.0)
    p.add_argument('--frame-dt', type=float, default=2.0)
    p.add_argument('--q-stab', type=float, default=0.005)
    p.add_argument('--sites', type=int, default=M.N_SITES)
    p.add_argument('--lanes', type=int, default=M.N_LANES)
    p.add_argument('--f-tracer', type=float, default=M.F_TRACER)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', required=True)
    A = p.parse_args()
    dens, trac, times, meta = run(A)
    os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
    np.savez_compressed(A.out, meta=json.dumps(meta), times=times, dens=dens, trac=trac,
                        cres=np.zeros((len(times), A.sites), np.uint16),
                        maxocc=np.ones(len(times), np.int32),
                        ntot=np.stack([dens.sum((1, 2)) - trac.sum((1, 2)), trac.sum((1, 2)),
                                       np.zeros(len(times))], 1).astype(np.int64))
    print('wrote %s: %d frames, %.1f s, final rho %.3f, jam (rho>1/2, last 100 sites) %.2f'
          % (A.out, len(times), meta['wall_s'], dens[-1].mean(), dens[-1, :, -100:].mean()))


if __name__ == '__main__':
    main()
