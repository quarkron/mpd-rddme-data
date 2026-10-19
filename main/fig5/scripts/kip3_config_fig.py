"""Configurations, trajectories and kymographs of the Kip3 traffic runs in one figure.

    python kip3_config_fig.py --exc out/fig1c_exc_iso_1500.npz --sw out/fig1c_exc_sw_1500.npz \
        --pt out/fig1c_pt_iso_1500.npz --times 0,400,800,1200 --out out/fig_config

WHAT IT DRAWS
    (a)-(d)  the configuration of the tube at each requested time: 16 lane rows against position,
             one mark per occupied site, so a single motor is one dot in the dilute region and the
             jam is a solid block. This is the object the lattice actually holds; the kymographs
             below are its lane average over time.
    (e)-(g)  kymographs of the normalized motor density for the three arms, with the snapshot
             times marked, and individual tracer trajectories drawn on the exclusion arm. A
             trajectory kinks where it enters the jam, which is the slowdown made visible.
    (h)      normalized density profiles at the same times.
    (i)      tracer speed against position at late times, with the profile faint behind it.

RHO
    The normalized motor density runs from 0 to 1: motors per binding site, normalized to the
    saturation of one motor per site that hard-core exclusion imposes. It is the quantity
    mean-field (1 - rho) is a statement about, Leduc's Langmuir density rho_s is quoted in, and
    panel (i)'s speeds are scored against, and it is comparable between arms without rescaling.
    Point particles have no such ceiling and reach 1.5 motors per voxel at the plus end, which
    saturates the colour axis; that overshoot is the argument, not an artefact. `--rho runmax`
    instead divides each arm by its own largest value, which spreads the point arm's tip spike
    over the whole scale and flattens everything else.

Runs in the lm2.6 container (numpy + matplotlib); jcp_style from studies/common.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M
import kip3_palette as P
from kip3_analyze import (load, kymo, link_tracers_full, style, SHORT, _key,
                          arm_handles, legend_row)


def rho_field(d, mode='site'):
    """(F, L) normalized motor density. 'site' = motors per binding site, normalized to the
    one-motor-per-site saturation, so it spans [0, 1] wherever exclusion is on. 'runmax' divides
    by the run's own largest value instead."""
    K = kymo(d)
    if mode == 'site':
        return K
    mx = float(K.max())
    return K / mx if mx > 0 else K


def tracks(d, window=40, min_len=6):
    """Chain tracer positions frame to frame within a lane by nearest forward match. Tracers are
    1.5 % of the motors, about three per lane, so candidates are hundreds of sites apart and the
    greedy match is unambiguous; a track ends when its tracer detaches or leaves at the tip."""
    trac, t = d['trac'], d['times']
    F, NL, L = trac.shape
    out = []
    for l in range(NL):
        active = [{'ts': [t[0]], 'ps': [int(p)]} for p in np.flatnonzero(trac[0, l])]
        for f in range(1, F):
            cur = set(int(x) for x in np.flatnonzero(trac[f, l]))
            active.sort(key=lambda tr: tr['ps'][-1])
            new, used = [], set()
            for tr in active:
                p = tr['ps'][-1]
                cands = sorted(c for c in cur if c not in used and p <= c <= p + window)
                if cands:
                    used.add(cands[0])
                    tr['ts'].append(t[f]); tr['ps'].append(cands[0]); new.append(tr)
                elif len(tr['ps']) >= min_len:
                    out.append(tr)
            for c in sorted(cur - used):
                new.append({'ts': [t[f]], 'ps': [c]})
            active = new
        out.extend(tr for tr in active if len(tr['ps']) >= min_len)
    return sorted(out, key=lambda tr: -len(tr['ps']))


def sample_tracks(trs, n, min_len=20):
    """Pick n tracks spread over start time. Sorting by length alone selects only late tracks:
    the long ones are those that end up parked in the jam, while a tracer that arrives early
    walks to the tip and leaves within ~200 s."""
    ok = [tr for tr in trs if len(tr['ps']) >= min_len] or trs
    ok = sorted(ok, key=lambda tr: tr['ts'][0])
    if len(ok) <= n:
        return ok
    idx = np.linspace(0, len(ok) - 1, n).round().astype(int)
    return [ok[i] for i in idx]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--exc', required=True); p.add_argument('--sw'); p.add_argument('--pt')
    p.add_argument('--times', default='0,400,800,1200')
    p.add_argument('--config-arm', choices=('exc', 'sw', 'pt'), default='exc')
    p.add_argument('--rho', choices=('site', 'runmax'), default='site')
    p.add_argument('--vmax', type=float, default=1.0, help='colour ceiling')
    p.add_argument('--n-tracks', type=int, default=30)
    p.add_argument('--window', type=int, default=40)
    p.add_argument('--w', type=int, default=10)
    p.add_argument('--t-min', type=float, default=1200.0, help='late window for panel (i)')
    p.add_argument('--out', required=True)
    A = p.parse_args()

    jcp = style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = [(k, load(getattr(A, k))) for k in ('pt', 'exc', 'sw') if getattr(A, k)]
    by_key = dict(runs)
    cols = dict(P.ARM)
    times = [float(x) for x in A.times.split(',')]
    vmax = A.vmax
    rlabel = (r'normalized motor density $\rho$' if A.rho == 'site'
              else r'normalized motor density $\rho/\rho_{\mathrm{max}}$')

    fig = plt.figure(figsize=(jcp.FULL, 7.9))
    gsA = fig.add_gridspec(len(times), 1, left=0.085, right=0.86, top=0.975, bottom=0.665,
                           hspace=0.55)
    gsB = fig.add_gridspec(1, len(runs), left=0.085, right=0.86, top=0.585, bottom=0.30,
                           wspace=0.14)
    gsC = fig.add_gridspec(1, 2, left=0.085, right=0.965, top=0.185, bottom=0.050, wspace=0.30)
    letters = 'abcdefghijkl'

    # ---- (a)-(d) configurations: 16 lanes x position, one mark per occupied site -------------
    cmeta, cd = by_key[A.config_arm]
    L = cmeta['sites']; xmax = L * M.H_NM * 1e-3
    for i, tt in enumerate(times):
        ax = fig.add_subplot(gsA[i, 0])
        fi = int(np.argmin(np.abs(cd['times'] - tt)))
        occ = cd['dens'][fi].astype(float)                      # (16, L), lanes x sites
        ax.imshow(np.minimum(occ, 1.0), aspect='auto', origin='lower', cmap=P.cmap_binary(),
                  vmin=0, vmax=1, extent=[0, xmax, -0.5, occ.shape[0] - 0.5],
                  interpolation='nearest')
        ax.set_yticks([0, 15]); ax.set_yticklabels(['1', '16'], fontsize=7)
        ax.set_ylabel('lane', fontsize=7.5, labelpad=1)
        ax.tick_params(labelsize=7)
        if i < len(times) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'position along the microtubule ($\mu$m)')
        n = int(cd['dens'][fi].sum())
        note = '  (empty track; motors are added at $t$ = 0)' if n == 0 else ''
        ax.text(0.005, 1.10, '(%s)  $t$ = %.0f s,  %d motors bound%s'
                % (letters[i], cd['times'][fi], n, note), transform=ax.transAxes, fontsize=7.5)
    fig.text(0.875, 0.82, '%s\nconfigurations:\none mark = one\noccupied site' % SHORT[_key(cmeta)],
             fontsize=7.5, va='center', ha='left', color=cols[A.config_arm])

    # ---- (e)-(g) kymographs, snapshot times marked, trajectories on the exclusion arm --------
    for j, (k, (meta, d)) in enumerate(runs):
        ax = fig.add_subplot(gsB[0, j])
        R = rho_field(d, A.rho); t = d['times']
        im = ax.imshow(R, aspect='auto', origin='upper', cmap=P.cmap(), vmin=0, vmax=vmax,
                       extent=[0, xmax, t[-1], 0], interpolation='nearest')
        for tt in times:
            ax.axhline(tt, color='w', lw=0.5, ls=':', alpha=0.65)
        if k == A.config_arm:
            for tr in sample_tracks(tracks(d, window=A.window), A.n_tracks):
                ax.plot(np.array(tr['ps']) * M.H_NM * 1e-3, tr['ts'], '-', color='#f2ffb0',
                        lw=0.55, alpha=0.9)
            ax.text(0.05, 0.115, 'single-tracer\ntrajectories', transform=ax.transAxes,
                    color='#f2ffb0', fontsize=6.5)
        ax.set_title(SHORT[_key(meta)], fontsize=8)
        ax.set_xticks([0, 5, 10])
        if j == 0:
            ax.set_ylabel('time (s)')
        else:
            ax.set_yticklabels([])
        ax.text(0.05, 0.04, '(%s)' % letters[len(times) + j], transform=ax.transAxes,
                color='w', fontsize=9)
    fig.text(0.4725, 0.262, r'position along the microtubule ($\mu$m)', ha='center')
    cax = fig.add_axes([0.875, 0.31, 0.013, 0.265])
    fig.colorbar(im, cax=cax, label=rlabel)

    # ---- (h) profiles at the snapshot times --------------------------------------------------
    ax = fig.add_subplot(gsC[0, 0])
    s = (np.arange(L) + 0.5) * M.H_NM * 1e-3
    for k, (meta, d) in runs:
        R = rho_field(d, A.rho)
        for i, tt in enumerate(times):
            fi = int(np.argmin(np.abs(d['times'] - tt)))
            if R[fi].max() == 0:
                continue
            ax.plot(s, np.convolve(R[fi], np.ones(10) / 10, mode='same'), '-', color=cols[k],
                    lw=0.9, alpha=0.3 + 0.7 * (i + 1) / len(times),
                    label=SHORT[_key(meta)] if i == len(times) - 1 else None)
    ax.set_xlabel(r'position ($\mu$m)'); ax.set_ylabel(rlabel, fontsize=7.5)
    ax.set_xlim(0, xmax); ax.set_ylim(0, vmax * 1.12)
    ax.text(0.03, 0.93, '(%s)  faint to solid: $t$ = %s s'
            % (letters[len(times) + len(runs)], ', '.join('%.0f' % x for x in times)),
            transform=ax.transAxes, fontsize=6.5, ha='left', va='top')

    # ---- (i) tracer speed against position at late times ------------------------------------
    ax = fig.add_subplot(gsC[0, 1])
    axr = ax.twinx()
    nb = 25
    for k, (meta, d) in runs:
        v0 = meta['v0_nm_s']
        v, _rho, pos = link_tracers_full(d, window=A.window, w=A.w, t_min=A.t_min)
        edges = np.linspace(0, L, nb + 1)
        idx = np.clip(np.digitize(pos, edges) - 1, 0, nb - 1)
        xs, ys, es = [], [], []
        for b in range(nb):
            m = idx == b
            if m.sum() >= 30:
                xs.append(0.5 * (edges[b] + edges[b + 1]) * M.H_NM * 1e-3)
                ys.append(v[m].mean() / v0); es.append(v[m].std() / np.sqrt(m.sum()) / v0)
        ax.errorbar(xs, ys, yerr=es, fmt='o', ms=2.8, color=cols[k], capsize=1.2,
                    label=SHORT[_key(meta)])
        late = d['times'] >= A.t_min
        axr.plot(s, rho_field(d, A.rho)[late].mean(0), '-', color=cols[k], lw=0.7, alpha=0.3)
    ax.axhline(1.0, color=jcp.MUTED, lw=0.7, ls=':')
    ax.set_xlabel(r'position ($\mu$m)'); ax.set_ylabel(r'$v / v_0$')
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.2)
    axr.set_ylim(0, vmax * 1.12)
    axr.set_ylabel(r'$\rho$ (faint)', fontsize=7)
    legend_row(fig, arm_handles(runs, cols, jcp), 0.225, fontsize=6.5, x=0.5)
    ax.text(0.03, 0.93, '(%s)  $t \\geq$ %.0f s' % (letters[len(times) + len(runs) + 1], A.t_min),
            transform=ax.transAxes, fontsize=7, ha='left', va='top')

    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None)
    print('wrote %s.pdf/.png  (rho = %s, vmax %.1f)' % (A.out, A.rho, vmax))
    for k, (meta, d) in runs:
        tr = tracks(d, window=A.window)
        print('  %-4s %d tracks >= 6 frames, longest %d frames; motors bound at %s s: %s'
              % (k, len(tr), len(tr[0]['ps']) if tr else 0, A.times,
                 [int(d['dens'][int(np.argmin(np.abs(d['times'] - tt)))].sum()) for tt in times]))


if __name__ == '__main__':
    main()
