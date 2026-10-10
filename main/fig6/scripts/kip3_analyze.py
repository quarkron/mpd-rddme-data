"""Reduce Kip3 traffic runs to the observables of Leduc et al. 2012 and draw the figure.

    python kip3_analyze.py fig1c --exc out/fig1c_exc_iso_1500.npz --sw out/fig1c_exc_sw_1500.npz \
        --pt out/fig1c_pt_iso_1500.npz --ref out/ref_fig1c.npz --out out/fig_fig1c
    python kip3_analyze.py speed --exc out/fig3dhi_exc_iso.npz --pt out/fig3dhi_pt_iso.npz \
        --ref out/ref_fig3dhi.npz --out out/fig_speed

OBSERVABLES
    kymograph   lane-averaged occupancy rho(s, t), time down and position across, as TIRF shows it
    profiles    rho(s) at chosen times
    jam length  sites from the plus end whose 20-site-smoothed lane-averaged occupancy exceeds
                one half (Leduc: the extent of the high-density region)
    speed       tracer speed against the local density of OTHER motors, from tracer positions
                linked between consecutive frames. Linking is one-to-one within a lane inside a
                forward window of `--window` sites; tracers that attach, detach or find no unique
                partner are dropped. The local density is the occupancy of the 2w+1 sites around
                the tracer at the earlier frame, the tracer's own site excluded, as the mCherry
                channel around an EGFP molecule in the assay.

Runs in the lm2.6 container (numpy + matplotlib); jcp_style from studies/common.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kip3_model as M
import kip3_palette as P


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    return meta, {k: z[k] for k in z.files if k != 'meta'}


def kymo(d):
    """Lane-averaged occupancy (F, L)."""
    return d['dens'].mean(1)


def jam_length(d, smooth=20, thresh=0.5):
    la = kymo(d)
    k = np.ones(smooth) / smooth
    sm = np.array([np.convolve(row, k, mode='same') for row in la])
    return (sm > thresh).sum(1)


def link_tracers_full(d, window=15, w=10, t_min=0.0):
    """Link tracers between consecutive frames within a lane; return (v_nm_s, rho_local,
    site) per linked pair. `site` is the position at the earlier frame."""
    trac, dens, t = d['trac'], d['dens'], d['times']
    F, NL, L = trac.shape
    h = 10.0
    vs, rhos, ps = [], [], []
    for l in range(NL):
        for f in range(F - 1):
            if t[f] < t_min:
                continue
            a = np.flatnonzero(trac[f, l]); b = np.flatnonzero(trac[f + 1, l])
            if len(a) == 0 or len(b) == 0:
                continue
            used = np.zeros(len(b), bool)
            for i, p in enumerate(a):
                cand = np.flatnonzero((b >= p) & (b <= p + window) & ~used)
                if len(cand) != 1:
                    continue
                j = cand[0]
                back = np.flatnonzero((a <= b[j]) & (a >= b[j] - window))
                if len(back) != 1:
                    continue
                used[j] = True
                lo, hi = max(0, p - w), min(L, p + w + 1)
                occ_others = dens[f, l, lo:hi].sum() - dens[f, l, p]
                rhos.append(occ_others / float(hi - lo - 1))
                vs.append((b[j] - p) * h / (t[f + 1] - t[f]))
                ps.append(p)
    return np.array(vs), np.array(rhos), np.array(ps)


def link_tracers(d, window=15, w=10):
    v, rho, _ = link_tracers_full(d, window, w)
    return v, rho


def wave_velocity(d, s0, s1, lag=1, t_min=0.0):
    """Propagation velocity of density fluctuations in sites [s0, s1), from the shift that
    maximises the cross-correlation of the lane-averaged density between frames f and f+lag.
    This is the slope of the striations a kymograph shows; in an exclusion process it is the
    collective (kinematic) velocity v0 (1 - 2 rho), NOT the motor velocity, and it changes
    sign at rho = 1/2."""
    K = kymo(d); t = d['times']
    W = K[:, s0:s1]
    late = t >= t_min
    # Remove the STATIC profile, not just each frame's mean: a window containing the domain
    # wall or the antenna ramp carries a large immobile structure, and correlating on it pins
    # the peak to zero shift whatever the fluctuations do.
    W = W - W[late].mean(0)
    Mw = W.shape[1]
    shifts = np.arange(-Mw + 1, Mw)
    acc = np.zeros(len(shifts))
    n = 0
    for f in range(len(t) - lag):
        if t[f] < t_min:
            continue
        x = W[f] - W[f].mean(); y = W[f + lag] - W[f + lag].mean()
        if x.std() == 0 or y.std() == 0:
            continue
        acc += np.correlate(y, x, mode='full')      # index Mw-1+k -> sum_i y[i+k] x[i]
        n += 1
    if n == 0:
        return float('nan'), float('nan')
    acc /= n
    # parabolic refinement around the integer peak
    k = int(shifts[np.argmax(acc)])
    i = int(np.argmax(acc))
    if 0 < i < len(acc) - 1:
        y0, y1, y2 = acc[i - 1], acc[i], acc[i + 1]
        den = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    dt = t[lag] - t[0]
    # the density this velocity belongs to is the one over the SAME frames, not over the whole
    # run: averaging in the empty early track would report a jam window as half-filled
    return k * 10.0 / dt, K[late, s0:s1].mean()


def bin_speed(v, rho, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(rho, edges) - 1, 0, nb - 1)
    out = []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 20:
            out.append((0.5 * (edges[b] + edges[b + 1]), v[m].mean(), v[m].std() / np.sqrt(m.sum()),
                        int(m.sum())))
    return np.array(out)


def style(sans=False):
    """House geometry and type, matching the manuscript's own figure scripts; sans=True for the
    sans-serif variant (opt-in per figure, see jcp_style.sans).

    ★jcp_style.apply() alone leaves text.usetex off, so every label renders in DejaVu Serif
    mathtext while the other figures in this manuscript render in Computer Modern - the same
    silent fallback the ramp-figure notes warn about. bench_fig.py, rxl80_occ_fig.py and
    fig_ribosome_arms.py all probe for latex and switch usetex on; do the same here, and the
    fonts match. TeX Live is on the container's PATH via lmrun.sh."""
    import matplotlib, subprocess
    matplotlib.use('Agg')
    sys.path.insert(0, '/ftsz/studies/common')
    import jcp_style
    jcp_style.apply(categorical=True)
    try:
        subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
        tex = True
    except Exception:
        tex = False
    matplotlib.rcParams.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']}
                               if tex else {'mathtext.fontset': 'cm'})
    if sans:
        matplotlib.rcParams.update(jcp_style.sans(
            {'text.latex.preamble': matplotlib.rcParams.get('text.latex.preamble', '')}, tex))
    if not tex:
        print('WARNING: latex not found; labels fall back to mathtext and will NOT match the '
              'manuscript figures')
    return jcp_style


LABEL = {'excluded/isolated': 'exclusion, no sidestepping',
         'excluded/switching': 'exclusion, with sidestepping',
         'point/isolated': 'point particles', 'tasep_lk/isolated': '1-D TASEP-LK reference'}
SHORT = {'excluded/isolated': 'exclusion, no sidestepping',
         'excluded/switching': 'exclusion, with sidestepping',
         'point/isolated': 'point particles', 'tasep_lk/isolated': '1-D reference'}


def arm_handles(runs, cols, jcp, extra=None):
    """Proxy handles for one shared key. Series labels here are long enough that three of them
    will not fit across a half-width panel, so the key goes once per FIGURE, above the row, not
    once per axes: any in-axes placement eventually sits on data - the faint density profiles
    behind the speed panels are the usual casualty - and an opaque frame hides points rather
    than avoiding them."""
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=cols[k], marker='o', ms=3.0, lw=1.1, label=SHORT[_key(meta)])
          for k, (meta, _d) in runs]
    if extra:
        hs.append(Line2D([], [], color=jcp.MUTED, lw=0.9, label=extra))
    return hs


def legend_row(fig, handles, y, ncol=None, fontsize=7, x=0.53):
    leg = fig.legend(handles=handles, loc='center', bbox_to_anchor=(x, y),
                     ncol=ncol or len(handles), fontsize=fontsize, frameon=False,
                     handletextpad=0.5, columnspacing=1.6)
    return leg


def _key(meta):
    return '%s/%s' % (meta['arm'], meta['lanes'])


def fig_fig1c(A):
    jcp = style()
    import matplotlib.pyplot as plt
    runs = [(k, load(getattr(A, k))) for k in ('pt', 'exc', 'sw', 'ref') if getattr(A, k)]
    n = len(runs)
    cols = dict(P.ARM)
    fig = plt.figure(figsize=(jcp.FULL, 5.6))
    top = fig.add_gridspec(1, n, left=0.07, right=0.88, bottom=0.55, top=0.93, wspace=0.12)
    bot = fig.add_gridspec(1, 2, left=0.07, right=0.975, bottom=0.08, top=0.355,
                           wspace=0.46)   # room for (d)'s twin axis label
    letters = 'abcdefgh'
    for i, (k, (meta, d)) in enumerate(runs):
        ax = fig.add_subplot(top[0, i])
        K = kymo(d); t = d['times']; L = K.shape[1]
        im = ax.imshow(K, aspect='auto', origin='upper', cmap=P.cmap(), vmin=0, vmax=1,
                       extent=[0, L * M.H_NM * 1e-3, t[-1], 0], interpolation='nearest')
        ax.set_title(SHORT[_key(meta)], fontsize=8)
        if i == 0:
            ax.set_ylabel('time (s)')
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, 5, 10])
        ax.text(0.05, 0.04, '(%s)' % letters[i], transform=ax.transAxes, color='w', fontsize=9)
    fig.text(0.475, 0.475, r'position along the microtubule ($\mu$m)', ha='center', va='center')
    cax = fig.add_axes([0.90, 0.58, 0.012, 0.32])
    fig.colorbar(im, cax=cax, label=r'occupancy $\rho$')

    # (d) speed and density against position: where the motors slow, and where the wall is
    ax = fig.add_subplot(bot[0, 0])
    axr = ax.twinx()
    s_um = None
    for k, (meta, d) in runs:
        v0 = meta['v0_nm_s']; L = meta['sites']
        v, _r, pos = link_tracers_full(d, window=40, w=10, t_min=A.speed_t_min)
        edges = np.linspace(0, L, 26)
        idx = np.clip(np.digitize(pos, edges) - 1, 0, 24)
        xs, ys, es = [], [], []
        for bn in range(25):
            m = idx == bn
            if m.sum() >= 30:
                xs.append(0.5 * (edges[bn] + edges[bn + 1]) * M.H_NM * 1e-3)
                ys.append(v[m].mean() / v0); es.append(v[m].std() / np.sqrt(m.sum()) / v0)
        ax.errorbar(xs, ys, yerr=es, fmt='o', ms=2.8, color=cols[k], capsize=1.2)
        s_um = (np.arange(L) + 0.5) * M.H_NM * 1e-3
        axr.plot(s_um, kymo(d)[d['times'] >= A.speed_t_min].mean(0), '-', color=cols[k],
                 lw=0.7, alpha=0.32)
    ax.axhline(1.0, color=jcp.MUTED, lw=0.7, ls=':')
    ax.set_xlabel(r'position along the microtubule ($\mu$m)'); ax.set_ylabel(r'$v / v_0$')
    ax.set_xlim(0, s_um[-1] + 0.05); ax.set_ylim(0, 1.2)
    axr.set_ylim(0, 1.2); axr.set_ylabel(r'$\rho$', fontsize=7.5)
    # Panel letters follow the manuscript's own convention (bench_fig, rxl80_occ_fig,
    # fig_ribosome_arms): above the frame at the axes' left edge, where they clear the tick
    # labels because those sit to the LEFT of the axes rather than above it. The time window
    # goes in the canvas's top-right, the one strip both panels leave empty.
    ax.text(0.0, 1.02, '(%s)' % letters[n], transform=ax.transAxes, fontsize=8.5,
            ha='left', va='bottom')
    ax.text(0.975, 0.965, r'$t \geq$ %.0f s' % A.speed_t_min, transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top')

    # speed against the local density of other motors, from the same runs
    ax = fig.add_subplot(bot[0, 1])
    for k, (meta, d) in runs:
        v, rho, _ = link_tracers_full(d, window=40, w=10, t_min=A.speed_t_min)
        Bs = bin_speed(v, rho)
        if not len(Bs):
            continue
        v0 = meta['v0_nm_s']
        ax.errorbar(Bs[:, 0], Bs[:, 1] / v0, yerr=Bs[:, 2] / v0, fmt='o', ms=3.0, color=cols[k],
                    capsize=1.2, label=SHORT[_key(meta)])
    ax.set_xlabel(r'local motor density $\rho$'); ax.set_ylabel(r'$v / v_0$')
    ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.27)   # headroom so the note clears the top points
    ax.text(0.0, 1.02, '(%s)' % letters[n + 1], transform=ax.transAxes, fontsize=8.5,
            ha='left', va='bottom')
    ax.text(0.975, 0.965, r'$t \geq$ %.0f s' % A.speed_t_min, transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top')
    legend_row(fig, arm_handles(runs, cols, jcp), 0.415)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext), bbox_inches=None)
    print('wrote %s.pdf/.png' % A.out)
    for k, (meta, d) in runs:
        jl = jam_length(d) * M.H_NM * 1e-3
        print('  %-4s jam length at end %.2f um; rho first 500 sites %.3f, last 100 sites %.3f'
              % (k, jl[-1], kymo(d)[-1, :500].mean(), kymo(d)[-1, -100:].mean()))


def fig_speed(A):
    jcp = style()
    import matplotlib.pyplot as plt
    runs = [(k, load(getattr(A, k))) for k in ('exc', 'pt', 'ref') if getattr(A, k)]
    fig, ax = plt.subplots(figsize=(jcp.COL, 2.9))
    cols = dict(P.ARM)
    for k, (meta, d) in runs:
        v, rho = link_tracers(d, window=A.window, w=A.w)
        B = bin_speed(v, rho)
        v0 = meta['v0_nm_s']
        ax.errorbar(B[:, 0], B[:, 1] / v0, yerr=B[:, 2] / v0, fmt='o' if k != 'ref' else 's',
                    ms=3.2, color=cols[k], mfc='none' if k == 'ref' else cols[k], capsize=1.5,
                    label=LABEL[_key(meta)])
        print('  %-4s %d links; v/v0 by rho bin: %s'
              % (k, len(v), ' '.join('%.2f:%.2f' % (b[0], b[1] / v0) for b in B)))
    r = np.linspace(0, 1, 50)
    ax.plot(r, 1 - r, '-', color=jcp.MUTED, lw=0.9, label=r'mean field $1-\rho$')
    ax.set_xlabel(r'local density of other motors $\rho$'); ax.set_ylabel(r'$v / v_0$')
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.15)
    fig.subplots_adjust(top=0.82)
    legend_row(fig, arm_handles(runs, cols, jcp, r'mean field $1-\rho$'), 0.93, ncol=2,
               x=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext))
    print('wrote %s.pdf/.png' % A.out)


def fig_slowdown(A):
    """Motor speed along the track against the density profile, and the velocity of the
    density waves in the dilute and the jammed region."""
    jcp = style()
    import matplotlib.pyplot as plt
    runs = [(k, load(getattr(A, k))) for k in ('exc', 'sw', 'pt') if getattr(A, k)]
    cols = dict(P.ARM)
    fig, axes = plt.subplots(1, 2, figsize=(jcp.FULL, 2.9))
    ax, ax2 = axes
    axr = ax.twinx()
    nb = 25
    for k, (meta, d) in runs:
        L = meta['sites']; v0 = meta['v0_nm_s']
        v, rho, pos = link_tracers_full(d, window=A.window, w=A.w, t_min=A.t_min)
        edges = np.linspace(0, L, nb + 1)
        idx = np.clip(np.digitize(pos, edges) - 1, 0, nb - 1)
        xs, ys, es = [], [], []
        for b in range(nb):
            m = idx == b
            if m.sum() >= 30:
                xs.append(0.5 * (edges[b] + edges[b + 1]) * M.H_NM * 1e-3)
                ys.append(v[m].mean() / v0); es.append(v[m].std() / np.sqrt(m.sum()) / v0)
        ax.errorbar(xs, ys, yerr=es, fmt='o', ms=3.0, color=cols[k], capsize=1.2,
                    label=SHORT[_key(meta)])
        # density profile over the same late-time window, on the right axis
        t = d['times']; late = t >= A.t_min
        prof = kymo(d)[late].mean(0)
        s = (np.arange(L) + 0.5) * M.H_NM * 1e-3
        axr.plot(s, prof, '-', color=cols[k], lw=0.8, alpha=0.35)
        jl = jam_length(d) * M.H_NM * 1e-3
        late = d['times'] >= A.t_min
        v_wall = (float(np.polyfit(d['times'][late], jl[late], 1)[0]) * 1e3
                  if jl[late].max() > 0 else 0.0)          # um/s -> nm/s, sign: growth backward
        print('  %-4s %d links, t >= %.0f s; domain wall moves %+.2f nm/s (jam %.2f um)'
              % (k, len(v), A.t_min, -v_wall, jl[-1]))
        for name, (a, b) in (('dilute', A.ld), ('jam', A.hd)):
            m = (pos >= a) & (pos < b)
            vw, rw = wave_velocity(d, a, b, lag=1, t_min=A.t_min)
            if m.sum() >= 30:
                print('       %-6s sites %4d-%4d: rho %.3f  motor v %6.1f nm/s (%.2f v0)  '
                      'density wave %+7.1f nm/s  [v0(1-2rho) = %+7.1f]'
                      % (name, a, b, rw, v[m].mean(), v[m].mean() / v0, vw, v0 * (1 - 2 * rw)))
    ax.axhline(1.0, color=jcp.MUTED, lw=0.7, ls=':')
    ax.set_xlabel(r'position along the microtubule ($\mu$m)'); ax.set_ylabel(r'$v / v_0$')
    ax.set_ylim(0, 1.25); ax.set_xlim(0, 10)
    axr.set_ylabel(r'$\rho$ (faint)'); axr.set_ylim(0, 1.25)
    ax.text(0.03, 0.9, '(a)', transform=ax.transAxes, fontsize=8)

    # wave velocity against local density, against the collective velocity v0 (1 - 2 rho)
    for k, (meta, d) in runs:
        L = meta['sites']; v0 = meta['v0_nm_s']
        pts = []
        for a in range(0, L - A.win_w, A.win_w):
            vw, rw = wave_velocity(d, a, a + A.win_w, lag=1, t_min=A.t_min)
            if np.isfinite(vw):
                pts.append((rw, vw / v0))
        pts = np.array(pts)
        ax2.plot(pts[:, 0], pts[:, 1], 'o', ms=3.0, color=cols[k], label=SHORT[_key(meta)])
    r = np.linspace(0, 1, 50)
    ax2.plot(r, 1 - 2 * r, '-', color=jcp.MUTED, lw=0.9, label=r'$1-2\rho$ (collective)')
    ax2.axhline(0, color='0.8', lw=0.6)
    ax2.set_xlabel(r'local density $\rho$'); ax2.set_ylabel(r'density-wave velocity / $v_0$')
    ax2.set_xlim(0, 1); ax2.set_ylim(-1.3, 1.3)
    ax2.text(0.03, 0.9, '(b)', transform=ax2.transAxes, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    legend_row(fig, arm_handles(runs, cols, jcp, r'$1-2\rho$ (collective)'), 0.945, x=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig('%s.%s' % (A.out, ext))
    print('wrote %s.pdf/.png' % A.out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('fig1c'); a.set_defaults(func=fig_fig1c)
    a.add_argument('--exc'); a.add_argument('--sw'); a.add_argument('--pt'); a.add_argument('--ref')
    a.add_argument('--profile-times', default='500,1000,1500')
    a.add_argument('--speed-t-min', type=float, default=1200.0)
    a.add_argument('--out', required=True)
    b = sub.add_parser('speed'); b.set_defaults(func=fig_speed)
    b.add_argument('--exc'); b.add_argument('--pt'); b.add_argument('--ref')
    b.add_argument('--window', type=int, default=15); b.add_argument('--w', type=int, default=10)
    b.add_argument('--out', required=True)
    c = sub.add_parser('slowdown'); c.set_defaults(func=fig_slowdown)
    c.add_argument('--exc'); c.add_argument('--sw'); c.add_argument('--pt')
    c.add_argument('--window', type=int, default=15); c.add_argument('--w', type=int, default=10)
    c.add_argument('--t-min', type=float, default=800.0, help='use frames from this time on')
    c.add_argument('--ld', type=int, nargs=2, default=(100, 600), help='dilute site window')
    c.add_argument('--hd', type=int, nargs=2, default=(800, 990), help='jammed site window')
    c.add_argument('--win-w', type=int, default=50, help='width of the wave-velocity windows')
    c.add_argument('--out', required=True)
    A = p.parse_args()
    A.func(A)


if __name__ == '__main__':
    main()
