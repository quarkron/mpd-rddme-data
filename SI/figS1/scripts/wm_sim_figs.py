"""SI Figs. S1 (wm1_l2_convergence) and S4 (wm4_intravoxel_shape_h8, and _h4) from the SAVED runs of
wm_sim_campaign.py. Extracted 2026-10-01 from harmonic_trap.ipynb (cells 6, 8 and 13, unchanged except the
colours, the S4 key position, the output paths, and reading partA.npz / rep_s*.h5 in place of notebook
memory). Main-text palette (kip3_palette): data navy, guides INK_DIM; S4 the ordered ramp of WM3.

usage (lm2.6 container, TeX Live on PATH):  python wm_sim_figs.py
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

SIM = os.environ.get('WM_OUT', os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), 'wm_sim'))
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


# ======== WM1 (notebook cell 8) ========
# WM1: 1-D L2 convergence of the WPE-equilibrated marginal vs. the analytic
# Gaussian, as h -> 0. Plots the ABSOLUTE 1-D L2 error
#   L2(h) = || p(h) - a(h) ||_2
# where p is the time-averaged simulated x-marginal and a is the discrete
# analytic Gaussian on the same fine grid.
#
# Error bars are the THEORETICAL 1-SD Poisson shot-noise prediction:
#   E[||p-a||^2] = (1 - sum a^2) / n_part
#   SD[||p-a||]  = sqrt(2 sum a^2 / n_part^2) / (2 E[||p-a||])  (delta method)
# i.e. how much L2 would fluctuate across hypothetical replicates of the
# same n_part Monte Carlo simulation. A second dotted curve overlays the
# theoretical Poisson shot-noise mean -- at the current n_part the data
# falls right on top of it, confirming that what WM1 measures is the
# sampling floor, not the WPE quadrature error (which scales O(h^2.5)
# absolute, far below the floor; see WM5 for the WM prediction).
#
# O(h) and O(h^2) guide lines bracket the measurement as before.
# Saves figures/wm1_l2_convergence.{pdf,png}.
from matplotlib.ticker import FuncFormatter, NullFormatter
from pathlib import Path
import subprocess



# Probe for LaTeX once -- fall back to mathtext if absent.
try:
    subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
    USE_LATEX = True
except Exception:
    USE_LATEX = False
    print('[WM1] LaTeX binary not found; using matplotlib mathtext fallback.')

rc_pub = {
    'font.family':         'serif',
    'font.size':           14,
    'axes.labelsize':      16,
    'axes.titlesize':      15,
    'xtick.labelsize':     14,
    'ytick.labelsize':     14,
    'legend.fontsize':     14,
    'lines.linewidth':     1.6,
    'lines.markersize':    7,
    'axes.linewidth':      1.1,
    'xtick.direction':     'in',  'ytick.direction':     'in',
    'xtick.top':           True,  'ytick.right':         True,
    'xtick.minor.visible': True,  'ytick.minor.visible': True,
    'xtick.major.size':    6.0,   'ytick.major.size':    6.0,
    'xtick.minor.size':    3.0,   'ytick.minor.size':    3.0,
    'xtick.major.width':   1.0,   'ytick.major.width':   1.0,
    'xtick.minor.width':   0.8,   'ytick.minor.width':   0.8,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
}
if USE_LATEX:
    rc_pub.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']})
else:
    rc_pub.update({'mathtext.fontset': 'cm'})

print('  N |  h    | n_part |   abs L2 (1-D) |  shot mean   |   shot SD')
print('----+-------+--------+----------------+--------------+-------------')
h_arr, rel1_arr, rel3_arr, shot_mean_arr, shot_sd_arr = [], [], [], [], []
for N in Ns:
    h = L / N
    p = resA[N]['marg'] / resA[N]['marg'].sum()
    a = analytic_marginal(N)
    spp, saa, spa = (p * p).sum(), (a * a).sum(), (p * a).sum()
    abs1 = np.sqrt(np.sum((p - a)**2))                             # absolute 1-D L2
    abs3 = np.sqrt(max(spp**3 - 2.0 * spa**3 + saa**3, 0.0))        # absolute 3-D L2
    # Multinomial shot-noise prediction for ||p - a||_2:
    #   E[||p-a||^2] = (1 - sum a^2) / n_part
    #   Var[||p-a||^2] ~ 2 sum a^2 / n_part^2  (chi-squared, p_i << 1)
    #   SD[||p-a||]   = SD[||p-a||^2] / (2 E[||p-a||])  (delta method)
    n_part = resA[N]['n_part']
    sum_a2 = float(saa)
    shot_mean_L2  = np.sqrt(max(1.0 - sum_a2, 0.0) / n_part)
    shot_var_L2sq = 2.0 * sum_a2 / (n_part * n_part)
    shot_sd_L2    = np.sqrt(shot_var_L2sq) / (2.0 * shot_mean_L2) if shot_mean_L2 > 0 else 0.0
    h_arr.append(h); rel1_arr.append(abs1); rel3_arr.append(abs3)
    shot_mean_arr.append(shot_mean_L2); shot_sd_arr.append(shot_sd_L2)
    print(f'{N:4d}| {h:.3f} | {n_part:6d} |    {abs1:.3e}    |  {shot_mean_L2:.3e}  |  {shot_sd_L2:.2e}')
h_arr         = np.array(h_arr)
rel1_arr      = np.array(rel1_arr)
shot_mean_arr = np.array(shot_mean_arr)
shot_sd_arr   = np.array(shot_sd_arr)

with plt.rc_context(rc_pub):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    good = rel1_arr > 0
    ax.errorbar(h_arr[good], rel1_arr[good], yerr=shot_sd_arr[good],
                fmt='o-', color=C_DATA, mfc='white', mew=1.4,
                capsize=3, ecolor=C_DATA, elinewidth=1.0,
                label=r'measured 1-D $L^2$ error')
    # ax.plot(h_arr[good], shot_mean_arr[good], ':', color='C0', lw=1.6, alpha=0.85,
    #         label=r'Poisson shot-noise floor')
    # O(h) and O(h^2) guide lines, anchored at the coarsest h, with offsets so
    # they straddle the data and don't overlap each other at large h.
    ia   = int(np.argmax(h_arr[good])); h0 = h_arr[good][ia]; y0 = rel1_arr[good][ia]
    hl   = np.array([h_arr.min() / 1.4, h_arr.max() * 1.4])
    f    = np.sqrt(2.0)
    line_oh  = (y0 * f) * (hl / h0)**1
    line_oh2 = (y0 / f) * (hl / h0)**2
    ax.plot(hl, line_oh,  '--', color=C_GUIDE, lw=1.2)
    ax.plot(hl, line_oh2, ':', color=C_GUIDE, lw=1.6)

    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks(h_arr); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$h$', fontsize=20)
    ax.set_ylabel(r'$L^2$ error', fontsize=20)
    ax.grid(True, which='major', alpha=0.30, lw=0.5)
    ax.grid(True, which='minor', alpha=0.10, lw=0.4)
    #ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='0.6')

    # Inline slope labels rotated to match visual line angles
    fig.canvas.draw()
    def visual_slope_deg(ax, slope, x_ref, y_ref, x_step=1.10):
        x2 = x_ref * x_step
        y2 = y_ref * (x2 / x_ref)**slope
        p1 = ax.transData.transform((x_ref, y_ref))
        p2 = ax.transData.transform((x2, y2))
        return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))

    # Place each label at a point along its guide line, offset vertically
    h_lab_1 = h_arr.min() * 1.6;  y_lab_1 = (y0 * f) * (h_lab_1 / h0)**1
    h_lab_2 = h_arr.min() * 4.0;  y_lab_2 = (y0 / f) * (h_lab_2 / h0)**2
    ax.text(h_lab_1, y_lab_1 * 1.55, r'$\mathcal{O}(h)$', fontsize=20,
            ha='center', va='bottom', color=C_GUIDE,
            rotation=visual_slope_deg(ax, 1, h_lab_1, y_lab_1), rotation_mode='anchor')
    ax.text(h_lab_2, y_lab_2 * 0.4, r'$\mathcal{O}(h^2)$', fontsize=20,
            ha='center', va='bottom', color=C_GUIDE,
            rotation=visual_slope_deg(ax, 2, h_lab_2, y_lab_2), rotation_mode='anchor')

    if good.sum() >= 2:
        o, _ = np.polyfit(np.log(h_arr[good]), np.log(rel1_arr[good]), 1)
        print(f'\nabsolute 1-D L2 convergence order ~ {o:.2f}')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'pdf' / 'wm1_l2_convergence.pdf')
    fig.savefig(FIG_DIR / 'png' / 'wm1_l2_convergence.png', dpi=300)
    print('  -> figures/wm1_l2_convergence.{pdf,png}')


# ======== WM4 (notebook cell 13) ========
# WM4: empirical intra-voxel SHAPE with replicate + axis + frame averaging.
# VOXEL-MEAN NORMALIZATION: rho(x_v + delta) / <rho>_vox. Curves do NOT
# pass through 1 at delta = 0 in general -- the lift/dip there IS the WPE
# quadrature-error diagnostic, identical to what WM3-right shows directly
# via Delta(x_v) = (h^2 k / 24)(1 - k(x_v - c)^2).
#
# Two sampling modes, selected automatically based on data availability:
#
#   * 3-D mode (when repA[N_fine]['h5_paths'] exists):
#       For each (rep, frame, cardinal axis, parity sign), pull the
#       coarse-voxel slab from the HDF5 lazily. Within that slab, sum the
#       fine-grid bins WITHIN each perpendicular coarse voxel to build
#       N_c^2 coarse-voxel-local 1-D profiles. By isotropic separability
#       of the harmonic Gaussian, each such profile has the same expected
#       ratio shape, just independent shot noise.
#       n_sample per shell ~ n_rep * 6 cardinal * N_c^2 * n_frame
#         h_c=8, N_c=32 -> ~6144 per (rep,frame) -> ~552k per shell
#         h_c=4, N_c=64 -> ~24576 per (rep,frame) -> ~2.2M per shell
#       (empty-centroid voxels are masked out; effective n is smaller.)
#
#   * 1-D marginal mode (fallback when no h5_paths):
#       Use the 1-D x/y/z marginals from repA (or single-run resA).
#       n_sample per shell = n_rep * 3 axes * n_frame * 2 parity = up to 540.
#
# Error bars are THEORETICAL Poisson shot-noise SEM (not empirical
# std/sqrt(n_sample)). Per-sample Var(R[k]) for R = p[k]/<p>_vox via
# delta method on a multinomial ratio with shared-bin correlation:
#     Var(R[k])/R^2 = 1/(n_part*a[k]) - 1/(n_fc*n_part*<a>_vox)
# Same formula for all k inside the voxel -- the voxel-mean denominator
# weights all n_fc bins equally, unlike the centroid (which used only
# two). SEM = sqrt(Var/n_sample).
#
# This panel isolates the intra-voxel SHAPE (drift tilt + diffusive
# curvature) and validates that the WPE lattice produces the correct
# Boltzmann profile inside each coarse voxel. The WPE quadrature error
# (lift/dip) is no longer encoded here -- see WM3 right and WM5.
#
# Split into two separate figures, one per coarse spacing h_c. Same
# publication formatting as WM3.
# Saves figures/wm4_intravoxel_shape_h{8,4}.{pdf,png}.
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import subprocess, h5py



# Probe for LaTeX once -- fall back to mathtext if absent.
try:
    subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=5)
    USE_LATEX = True
except Exception:
    USE_LATEX = False
    print('[WM4] LaTeX binary not found; using matplotlib mathtext fallback.')

rc_pub = {
    'font.family':         'serif',
    'font.size':           14,
    'axes.labelsize':      16,
    'axes.titlesize':      15,
    'xtick.labelsize':     14,
    'ytick.labelsize':     14,
    'legend.fontsize':     14,
    'lines.linewidth':     1.6,
    'lines.markersize':    7,
    'axes.linewidth':      1.1,
    'xtick.direction':     'in',  'ytick.direction':     'in',
    'xtick.top':           True,  'ytick.right':         True,
    'xtick.minor.visible': True,  'ytick.minor.visible': True,
    'xtick.major.size':    6.0,   'ytick.major.size':    6.0,
    'xtick.minor.size':    3.0,   'ytick.minor.size':    3.0,
    'xtick.major.width':   1.0,   'ytick.major.width':   1.0,
    'xtick.minor.width':   0.8,   'ytick.minor.width':   0.8,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.05,
}
if USE_LATEX:
    rc_pub.update({'text.usetex': True, 'font.serif': ['Computer Modern Roman']})
else:
    rc_pub.update({'mathtext.fontset': 'cm'})

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

# Theoretical Poisson bars need (i) the analytic 1-D Gaussian at fine bins,
# and (ii) the effective per-sample particle count.
_a_fine_wm4 = analytic_marginal(N_fine)
if 'repA' in globals() and repA and N_fine in repA:
    _n_part_per_sample = float(np.mean(repA[N_fine]['n_part']))
else:
    _n_part_per_sample = float(resA[N_fine]['n_part'])
print(f'Poisson SEM uses n_part_per_sample = {_n_part_per_sample:.0f}')

print()
print('  h_c  | |x_v-c| | n_samp  | mean@delta=h_fine/2 | SEM    | mean@delta=+h_c/2 | SEM')
print('-------+---------+---------+---------------------+--------+-------------------+-------')
with plt.rc_context(rc_pub):
    for h_c in (8.0, 4.0):
        n_fc = int(round(h_c / h_fine))
        if n_fc < 2 or n_fc % 2 != 0:
            raise ValueError(f'n_fc={n_fc} must be even and >=2 for centroid interpolation')
        d_out = ((np.arange(n_fc) + 0.5) / n_fc - 0.5) * h_c

        # One pass through the data per h_c to grab all R shells.
        shells = _extract_3d_shells(h_c, R_LIST) if USE_3D else _extract_1d_shells(h_c, R_LIST)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for R, color in zip(R_LIST, COLORS):
            per_sample = shells[R]
            x_v_mag = abs(shell_voxels(h_c, R)[0][1])
            if per_sample.shape[0] == 0:
                print(f'  WARNING: no samples in shell |x_v-c|={x_v_mag:.1f}')
                continue
            mean = per_sample.mean(axis=0)
            # Theoretical Poisson shot-noise SEM (replaces empirical std/sqrt(n)).
            # Per-sample n_part = total particles in one (rep, axis, frame) marginal.
            sem  = _poisson_sem(h_c, R, _a_fine_wm4, _n_part_per_sample,
                                per_sample.shape[0])
            ax.errorbar(d_out / h_c, mean, yerr=sem, fmt='o-', ms=4, lw=1.0,
                        color=color, capsize=2, ecolor=color, elinewidth=0.9,
                        label=fr'$|x_v-c|={x_v_mag:.1f}$')
            # Taylor voxel-mean-normalized: rho_t / <rho_t>_vox
            # (midpoint Riemann over the n_fc fine bins; matches the empirical
            # voxel mean above byte-for-byte).
            phip = K * x_v_mag; phipp = K
            rho_t = 1 - phip*d_out - 0.5*phipp*d_out**2 + 0.5*(phip*d_out)**2
            ax.plot(d_out / h_c, rho_t / rho_t.mean(), '--', lw=1.0, alpha=0.7, color=color)
            kmid  = n_fc // 2
            kedge = n_fc - 1
            print(f'  {h_c:4.1f} | {x_v_mag:6.2f}  | {per_sample.shape[0]:7d} '
                  f'|     {mean[kmid]:.5f}      | {sem[kmid]:.1e} '
                  f'|     {mean[kedge]:.5f}     | {sem[kedge]:.1e}')
        ax.axhline(1.0, color=C_GUIDE, alpha=0.5, lw=0.7)
        ax.set_xlabel(r'$\delta / h_{\rm coarse}$', fontsize=20)
        ax.set_ylabel(r'$\rho(x_v+\delta) / \langle\rho\rangle_{\rm vox}$', fontsize=20)
        ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend(fontsize=14, loc='upper right', frameon=True, framealpha=0.95, edgecolor='0.6')   # every curve is below 1.0 there
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_pdf = FIG_DIR / 'pdf' / f'wm4_intravoxel_shape_h{h_c:g}.pdf'
        out_png = FIG_DIR / 'png' / f'wm4_intravoxel_shape_h{h_c:g}.png'
        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, bbox_inches='tight', dpi=300)
        print(f'  -> {out_pdf.name}, {out_png.name}')
