"""SI Figs. S1 (wm1_l2_convergence) and S4b (wm4_intravoxel_shape_h8, and the extra _h4) from the SAVED runs of
wm_sim_campaign.py. Extracted 2026-10-01 from harmonic_trap.ipynb (cells 6, 8 and 13, reading partA.npz /
rep_s*.h5 in place of notebook memory). Main-text palette (kip3_palette): data navy, guides INK_DIM; S4 the ordered
ramp of WM3.

2026-10-19: redrawn at final size in the publication style (FIGURE_STYLE.md section 5, common/jcp_style.py PUB_RC):
Arial text and math, 10/11 pt, no equations inside the panels (the order-of-convergence guides are named in the key;
the definitions are the caption's). In SI/figS4 this script also writes the combined fig:wm-intravoxel,
plots/wm_intravoxel.{pdf,png} (a = the closed form, drawn by wm_closedform_figs.draw_wm3; b = the h = 8 measurement),
and its panels.

usage:  python wm_sim_figs.py      # -> ../plots/{pdf,png}/wm1_*, wm4_* (+ ../plots/wm_intravoxel.* in SI/figS4)
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, MaxNLocator
from pathlib import Path
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'common'))
import jcp_style as J                               # noqa: E402
import panel_export                                 # noqa: E402

SIM = os.environ.get('WM_OUT', os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figS1', 'data'), 'wm_sim'))
FIG_DIR = Path(os.environ.get('WM_FIGD', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')))
L, K = 256.0, 0.01
SIGMA = 1.0 / np.sqrt(K)
N_SWEEP = [32, 64, 128, 256, 512]
STOPS = ['#050b23', '#0e2159', '#17509b', '#2093b4', '#41c5bd', '#a3dda0', '#e9f7bb']   # kip3_palette
C_DATA, C_PT, C_GUIDE = '#123f7a', '#9cbf3c', '#5d7185'

_z = np.load(os.path.join(SIM, 'partA.npz'))
resA = {int(N): dict(marg=_z['marg_%d' % N], n_part=int(_z['npart_%d' % N])) for N in _z['N_SWEEP']}
Ns = [N for N in N_SWEEP if N in resA]
_h5 = sorted(glob.glob(os.path.join(SIM, 'rep_s*.h5')))
repA = {}
if _h5:
    repA[max(Ns)] = dict(h5_paths=_h5,
                         n_part=np.array([h5py.File(p, 'r')['N%d' % max(Ns)].attrs['n_part'] for p in _h5]))

def analytic_marginal(N):
    h = L / N
    x = (np.arange(N) + 0.5 - N / 2.0) * h        # box-unit offset from center
    g = np.exp(-x**2 / (2 * SIGMA**2)); return g / g.sum()

def fit_sigma(marg, h):
    p = marg / marg.sum(); x = np.arange(len(marg))
    m = (x * p).sum(); var = (p * (x - m)**2).sum()
    return np.sqrt(var) * h                       # back to box units




def save(fig, stem):
    """Write ../plots/pdf/<stem>.pdf and ../plots/png/<stem>.png and run the publication checks."""
    pdf = FIG_DIR / 'pdf' / (stem + '.pdf')
    fig.savefig(pdf)
    fig.savefig(FIG_DIR / 'png' / (stem + '.png'), dpi=300)
    J.pub_check(fig, stem, si=True)
    J.check_text_overlap(fig, stem)
    J.pub_fonts(str(pdf), stem)
    print('  -> %s.{pdf,png} (%.2f x %.2f in)' % (stem, fig.get_figwidth(), fig.get_figheight()))


# ======== WM1 (notebook cell 8) ========
# WM1: 1-D L2 convergence of the WPE-equilibrated marginal vs. the analytic Gaussian, as h -> 0. Plots the ABSOLUTE
# 1-D L2 error L2(h) = || p(h) - a(h) ||_2, p the time-averaged simulated x-marginal and a the discrete analytic
# Gaussian on the same grid. Error bars are the THEORETICAL 1-SD Poisson shot-noise prediction:
#   E[||p-a||^2] = (1 - sum a^2) / n_part,  SD[||p-a||] = sqrt(2 sum a^2 / n_part^2) / (2 E[||p-a||])  (delta method)
# At the current n_part the data falls on the shot-noise mean: WM1 measures the sampling floor, not the WPE
# quadrature error (O(h^2.5) absolute, far below the floor; see WM5). O(h) and O(h^2) guides bracket the data.
def wm1_figure():
    print('  N |  h    | n_part |   abs L2 (1-D) |  shot mean   |   shot SD')
    h_arr, rel1_arr, shot_sd_arr = [], [], []
    for N in Ns:
        h = L / N
        p = resA[N]['marg'] / resA[N]['marg'].sum()
        a = analytic_marginal(N)
        abs1 = np.sqrt(np.sum((p - a)**2))                             # absolute 1-D L2
        n_part = resA[N]['n_part']
        sum_a2 = float((a * a).sum())
        shot_mean_L2 = np.sqrt(max(1.0 - sum_a2, 0.0) / n_part)
        shot_sd_L2 = np.sqrt(2.0 * sum_a2 / (n_part * n_part)) / (2.0 * shot_mean_L2) if shot_mean_L2 > 0 else 0.0
        h_arr.append(h); rel1_arr.append(abs1); shot_sd_arr.append(shot_sd_L2)
        print(f'{N:4d}| {h:.3f} | {n_part:6d} |    {abs1:.3e}    |  {shot_mean_L2:.3e}  |  {shot_sd_L2:.2e}')
    h_arr, rel1_arr, shot_sd_arr = np.array(h_arr), np.array(rel1_arr), np.array(shot_sd_arr)
    good = rel1_arr > 0
    if good.sum() >= 2:
        o, _ = np.polyfit(np.log(h_arr[good]), np.log(rel1_arr[good]), 1)
        print(f'\nabsolute 1-D L2 convergence order ~ {o:.2f}')

    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(4.0, 2.9)
        ax = J.pub_ax(fig, 0.78, 0.12, 3.12, 2.32)
        ax.errorbar(h_arr[good], rel1_arr[good], yerr=shot_sd_arr[good], fmt='o-', color=C_DATA, mfc='white',
                    mew=1.2, capsize=3, ecolor=C_DATA, elinewidth=1.0, label='measured')
        # O(h) and O(h^2) guides anchored at the coarsest h, offset so they straddle the data
        ia = int(np.argmax(h_arr[good])); h0 = h_arr[good][ia]; y0 = rel1_arr[good][ia]
        hl = np.array([h_arr.min() / 1.4, h_arr.max() * 1.4])
        f = np.sqrt(2.0)
        ax.plot(hl, (y0 * f) * (hl / h0)**1, '--', color=C_GUIDE, lw=1.2, label='first order')
        ax.plot(hl, (y0 / f) * (hl / h0)**2, ':', color=C_GUIDE, lw=1.6, label='second order')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(h_arr); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(hl[0], hl[1])
        ax.set_xlabel(r'lattice spacing $h$')
        ax.set_ylabel('$L$\u00b2 error')
        ax.grid(True, which='major', alpha=0.25, lw=0.5)
        hs, ls = ax.get_legend_handles_labels()
        ax.legend(hs[-1:] + hs[:-1], ls[-1:] + ls[:-1], loc='lower right')     # the measurement first
        save(fig, 'wm1_l2_convergence')
        panel_export.export(fig, 'wm1_l2_convergence')        # one panel: nothing is written
    return fig


# ======== WM4 (notebook cell 13) ========
# WM4: empirical intra-voxel SHAPE with replicate + axis + frame averaging, VOXEL-MEAN normalized
# rho(x_v + delta) / <rho>_vox. Two sampling modes: 3-D coarse-voxel-local profiles from the HDF5 frames (each
# (rep, frame, cardinal axis, parity sign, perpendicular coarse voxel) gives one profile) when repA has h5_paths,
# else the 1-D marginals. Error bars are the THEORETICAL Poisson shot-noise SEM of the ratio,
#     Var(R[k])/R^2 = 1/(n_part*a[k]) - 1/(n_fc*n_part*<a>_vox),  SEM = sqrt(Var/n_sample).
# One figure per coarse spacing h_c; the manuscript uses h_c = 8.
N_fine = max(Ns); h_fine = L / N_fine

USE_3D = ('repA' in globals() and repA and N_fine in repA
          and 'h5_paths' in repA[N_fine] and repA[N_fine]['h5_paths'])

if USE_3D:
    h5_paths = repA[N_fine]['h5_paths']
    n_part_arr = repA[N_fine]['n_part']
    src_label = f'3-D coarse-voxel-local: {len(h5_paths)} reps x N_c^2 perp positions x 6 cardinal'
elif 'repA' in globals() and repA and N_fine in repA:
    M = repA[N_fine]['marg']
    n_part = repA[N_fine]['n_part'][:, None, None]
    P_ens = (M / n_part[..., None]).reshape(-1, N_fine)
    src_label = (f'1-D marginal: {M.shape[0]} reps x {M.shape[1]} axes x {M.shape[2]} frames '
                 f'= {P_ens.shape[0]} samples (each split into 2 parity voxels)')
else:
    P_ens = (resA[N_fine]['marg'] / resA[N_fine]['marg'].sum())[None, :]
    src_label = '1-D marginal: single time-averaged run (1 sample, 2 parity voxels)'
print('source:', src_label)


def shell_voxels(h_c, R, eps=1e-6):
    """Coarse voxels at signed offset +/-R from c (straddling pair when R=0)."""
    N_c = int(round(L / h_c))
    targets = [+eps, -eps] if R == 0 else [+R, -R]
    seen, out = set(), []
    for t in targets:
        j = int(np.floor((t + L/2) / h_c)); j = min(max(j, 0), N_c - 1)
        if j in seen: continue
        seen.add(j)
        x_v = (j + 0.5 - N_c/2.0) * h_c
        out.append((j, x_v))
    return out


def _poisson_sem(h_c, R, a_fine, n_part_per_sample, n_sample, eps=1e-6):
    """Theoretical per-bin SEM of the VOXEL-MEAN-normalized ratio
    R[k] = p[k] / <p>_vox  under multinomial / Poisson shot noise.

    Delta method on a ratio of correlated multinomials yields:
      Var(R[k]) / R[k]^2  =  1/(n_part * a[k])  -  1/(n_fc * n_part * <a>_vox)

    Same formula for all k inside the voxel (no special mid-1, mid case).
    Returns (n_fc,) array of SEM values."""
    n_fc = int(round(h_c / h_fine))
    N_c  = int(round(L / h_c))
    targets = [+eps] if R == 0 else [+R]
    j_c     = int(np.floor((targets[0] + L/2) / h_c)); j_c = min(max(j_c, 0), N_c - 1)
    x_v     = (j_c + 0.5 - N_c/2.0) * h_c
    i0, i1  = j_c * n_fc, j_c * n_fc + n_fc
    a_voxel = a_fine[i0:i1].astype(np.float64)
    if x_v < 0:
        a_voxel = a_voxel[::-1]
    a_mean  = a_voxel.mean()
    if a_mean <= 0:
        return np.zeros(n_fc)
    R_true  = a_voxel / a_mean
    inv_nA  = 1.0 / (n_part_per_sample * np.where(a_voxel > 0, a_voxel, 1e-30))
    inv_nM  = 1.0 / (n_fc * n_part_per_sample * a_mean)
    var_one = np.zeros(n_fc)
    for k in range(n_fc):
        if a_voxel[k] <= 0:
            var_one[k] = np.inf
            continue
        term = inv_nA[k] - inv_nM
        var_one[k] = R_true[k]**2 * max(term, 0.0)
    n_eff = max(int(n_sample), 1)
    return np.sqrt(var_one / n_eff)



def _extract_3d_shells(h_c, R_list, eps=1e-6):
    """Stream each replicate's HDF5 frames once and harvest coarse-voxel-local
    profiles for every (R, cardinal direction, parity voxel, perpendicular
    coarse position). Returns dict {R: (n_total, n_fc) profiles array} of
    centroid-normalized intra-voxel profiles, oriented outward (+delta).
    """
    n_fc = int(round(h_c / h_fine))
    N_c  = int(round(L / h_c))
    # Per-R list of unique (axis, j_c, x_v) shell voxels.
    per_R_voxels = {}
    for R in R_list:
        tgts = [+eps, -eps] if R == 0 else [+R, -R]
        triples = []
        for axis_idx in range(3):
            seen = set()
            for t in tgts:
                j = int(np.floor((t + L/2) / h_c)); j = min(max(j, 0), N_c - 1)
                if j in seen: continue
                seen.add(j)
                x_v = (j + 0.5 - N_c/2.0) * h_c
                triples.append((axis_idx, j, x_v))
        per_R_voxels[R] = triples

    bucket = {R: [] for R in R_list}
    for h5_path in h5_paths:
        with h5py.File(h5_path, 'r') as f:
            ds = f[f'N{N_fine}/frames']               # (n_frame, N, N, N) uint8
            n_frame = ds.shape[0]
            for fi in range(n_frame):
                vol = ds[fi]                          # (N, N, N) uint8
                for R in R_list:
                    for axis_idx, j_c, x_v in per_R_voxels[R]:
                        i0 = j_c * n_fc; i1 = i0 + n_fc
                        if axis_idx == 0:
                            slab = vol[i0:i1, :, :]                       # (n_fc, N, N)
                        elif axis_idx == 1:
                            slab = vol[:, i0:i1, :].transpose(1, 0, 2)    # (n_fc, N, N)
                        else:
                            slab = vol[:, :, i0:i1].transpose(2, 0, 1)    # (n_fc, N, N)
                        # Sum fine perpendicular bins within each perpendicular coarse voxel.
                        s5 = slab.reshape(n_fc, N_c, n_fc, N_c, n_fc)
                        coarse_perp = s5.sum(axis=(2, 4))                 # (n_fc, N_c, N_c)
                        # Each (perp_a, perp_b) coarse cell -> one length-n_fc profile.
                        profs = coarse_perp.transpose(1, 2, 0).reshape(-1, n_fc).astype(np.float64)
                        if x_v < 0:
                            profs = profs[:, ::-1]                        # mirror outward
                        # Voxel-mean denominator (matches closed-form <rho>_vox).
                        p_vmean = profs.mean(axis=1)
                        ok = p_vmean > 0                                  # drop empty voxels
                        if not np.any(ok):
                            continue
                        bucket[R].append(profs[ok] / p_vmean[ok, None])
    return {R: (np.concatenate(rows, axis=0) if rows else np.zeros((0, n_fc)))
            for R, rows in bucket.items()}


def _extract_1d_shells(h_c, R_list):
    """1-D marginal fallback: stack parity voxels as separate samples."""
    n_fc = int(round(h_c / h_fine))
    out = {}
    for R in R_list:
        voxels = shell_voxels(h_c, R)
        rows = []
        for j, x_v in voxels:
            i0 = j * n_fc; i1 = i0 + n_fc
            p_slice = P_ens[:, i0:i1]
            if x_v < 0:
                p_slice = p_slice[:, ::-1]
            # Voxel-mean denominator (matches closed-form <rho>_vox).
            p_vmean = p_slice.mean(axis=1, keepdims=True)
            p_vmean = np.where(p_vmean > 0, p_vmean, 1.0)
            rows.append(p_slice / p_vmean)
        out[R] = np.concatenate(rows, axis=0) if rows else np.zeros((0, n_fc))
    return out


R_LIST = (0.0, SIGMA, 2*SIGMA)
COLORS = (STOPS[1], STOPS[3], C_PT)   # main-text palette, as WM3: ordered distance from the centre
R_LABELS = ('centre subvolume', r'subvolume holding $\sigma$', r'subvolume holding $2\sigma$')

# Theoretical Poisson bars need (i) the analytic 1-D Gaussian at fine bins,
# and (ii) the effective per-sample particle count.
_a_fine_wm4 = analytic_marginal(N_fine)
if 'repA' in globals() and repA and N_fine in repA:
    _n_part_per_sample = float(np.mean(repA[N_fine]['n_part']))
else:
    _n_part_per_sample = float(resA[N_fine]['n_part'])
print(f'Poisson SEM uses n_part_per_sample = {_n_part_per_sample:.0f}')


def wm4_data(h_c):
    """Per shell: (|x_v - c|, delta/h_c, mean profile, Poisson SEM, Taylor prediction)."""
    n_fc = int(round(h_c / h_fine))
    if n_fc < 2 or n_fc % 2 != 0:
        raise ValueError(f'n_fc={n_fc} must be even and >=2 for centroid interpolation')
    d_out = ((np.arange(n_fc) + 0.5) / n_fc - 0.5) * h_c
    shells = _extract_3d_shells(h_c, R_LIST) if USE_3D else _extract_1d_shells(h_c, R_LIST)
    out = []
    for R in R_LIST:
        per_sample = shells[R]
        x_v_mag = abs(shell_voxels(h_c, R)[0][1])
        if per_sample.shape[0] == 0:
            print(f'  WARNING: no samples in shell |x_v-c|={x_v_mag:.1f}')
            out.append(None)
            continue
        mean = per_sample.mean(axis=0)
        sem = _poisson_sem(h_c, R, _a_fine_wm4, _n_part_per_sample, per_sample.shape[0])
        # Taylor voxel-mean-normalized rho_t / <rho_t>_vox (midpoint Riemann over the n_fc fine bins)
        phip = K * x_v_mag; phipp = K
        rho_t = 1 - phip*d_out - 0.5*phipp*d_out**2 + 0.5*(phip*d_out)**2
        kmid, kedge = n_fc // 2, n_fc - 1
        print(f'  {h_c:4.1f} | {x_v_mag:6.2f}  | {per_sample.shape[0]:7d} '
              f'|     {mean[kmid]:.5f}      | {sem[kmid]:.1e} '
              f'|     {mean[kedge]:.5f}     | {sem[kedge]:.1e}')
        out.append((x_v_mag, d_out / h_c, mean, sem, rho_t / rho_t.mean()))
    return out


def draw_wm4(ax, shells):
    """Panel b of fig:wm-intravoxel: the measured intra-voxel density (markers) against the Taylor
    prediction (dashed), three subvolumes."""
    for sh, color in zip(shells, COLORS):
        if sh is None:
            continue
        _, x, mean, sem, pred = sh
        ax.errorbar(x, mean, yerr=sem, fmt='o-', ms=3.5, lw=1.0, color=color, capsize=2, ecolor=color,
                    elinewidth=0.9)
        ax.plot(x, pred, '--', lw=1.0, alpha=0.8, color=color)
    ax.axhline(1.0, color=C_GUIDE, alpha=0.5, lw=0.7)
    ax.set_xlabel(r'offset in the subvolume $\delta / h_{\mathrm{coarse}}$')
    ax.set_ylabel(r'$\rho(x_v+\delta)$ / subvolume mean')
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_xlim(-0.55, 0.55)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.25, lw=0.5)
    pos = ax.legend([Line2D([], [], color=c, marker='o', ms=3.5, lw=1.0) for c in COLORS], R_LABELS,
                    loc='upper right')
    ax.add_artist(pos)
    ax.legend([Line2D([], [], color=C_GUIDE, marker='o', ms=3.5, lw=1.0), Line2D([], [], color=C_GUIDE, ls='--')],
              ['measured', 'Taylor prediction'], loc='lower left')


PANEL_W, PANEL_H = 3.2, 2.9         # in, as wm_closedform_figs: one fig:wm-intravoxel panel on its own
PANEL_AX = (0.62, 0.12, 2.5, 2.28)


def wm4_figure(h_c, shells):
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(PANEL_W, PANEL_H)
        draw_wm4(J.pub_ax(fig, *PANEL_AX), shells)
        save(fig, f'wm4_intravoxel_shape_h{h_c:g}')
    return fig


def intravoxel_combined(shells8):
    """fig:wm-intravoxel as one file: a = the closed form (wm_closedform_figs.draw_wm3), b = the h = 8 measurement."""
    from wm_closedform_figs import draw_wm3
    W, H = 6.5, 2.95
    XA, XB, AW, AH, YT = 0.62, 3.88, 2.5, 2.2, 0.32
    with plt.rc_context(J.PUB_RC):
        fig = J.si_page(W, H)
        draw_wm3(J.pub_ax(fig, XA, YT, AW, AH))
        draw_wm4(J.pub_ax(fig, XB, YT, AW, AH), shells8)
        J.pub_letter(fig, 'a', 0.02, 0.02)
        J.pub_letter(fig, 'b', XB - 0.60, 0.02)
        pdf = FIG_DIR / 'wm_intravoxel.pdf'
        fig.savefig(pdf)
        fig.savefig(FIG_DIR / 'wm_intravoxel.png', dpi=300)
        J.pub_check(fig, 'wm_intravoxel', si=True)
        J.check_text_overlap(fig, 'wm_intravoxel')
        J.pub_fonts(str(pdf), 'wm_intravoxel')
        print('  -> wm_intravoxel.{pdf,png} (%.2f x %.2f in)' % (W, H))
        panel_export.export(fig, 'wm_intravoxel')
    return fig


def main():
    wm1_figure()
    print('\n  h_c  | |x_v-c| | n_samp  | mean@delta=h_fine/2 | SEM    | mean@delta=+h_c/2 | SEM')
    shells = {h_c: wm4_data(h_c) for h_c in (8.0, 4.0)}
    for h_c, sh in shells.items():
        wm4_figure(h_c, sh)
    if os.path.basename(os.path.dirname(HERE)) == 'figS4':     # S4 holds both halves of fig:wm-intravoxel
        intravoxel_combined(shells[8.0])


if __name__ == '__main__':
    main()
