"""Loaders for the sinusoidal-field reaction figure: profiles, counts, divergences, voxel fields.

Everything the figure draws comes from runs already on disk under studies/lmrddme_validation:

    ../mpd_crddme/rxl80c_out/{veryearly,early,late}_{excl,ideal}_s*.npz   the lattice arms
    ../readdy_ref/out_field/rf_{vearly,early,}s*.npz                      the iPRD reference
    ../kl3way_cache.npz                                                  the divergence curves

THE FIELD AXIS IS AXIS 0 OF THE RESHAPED OCCUPANCY.
    vex_reactive stores `occA`/`occC` as the C-order ravel of the HDF5 lattice with its particle
    axis summed away, and the solver's flat site index maps onto that shape so that the ramp lands
    on the FIRST axis -- the same convention `prof3[:, 0, :]` records. Guessing the other way
    averages over the ramp and returns a flat profile, so `field_axis_check` asserts it from the
    data rather than trusting the comment.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = os.path.join(HERE, '..', '..', '..', 'main', 'fig3', 'data', 'mpd_crddme', 'rxl80c_out')
RDY = os.path.join(HERE, '..', '..', '..', 'main', 'fig3', 'data', 'readdy_ref', 'out_field')
KL = os.path.join(HERE, '..', '..', '..', 'main', 'fig3', 'data', 'kl3way_cache.npz')

N = 32                # voxels per side
NPER = 8              # voxels per period of the field: lambda = 80 nm at h = 10 nm
TAU_SS = 3.0          # frames at tau >= this enter the steady profile
SPECIES = {'A': 0, 'B': 1, 'C': 2}


def meta(path):
    return json.loads(str(np.load(path, allow_pickle=True)['meta']))


def iprd_profile(name):
    """(phase, relative density, SEM over seeds) for species `name`, on the lattice's 10 nm voxels.

    The reference's raw 5 nm bins are summed pairwise into 10 nm windows CENTERED on the lattice
    field nodes, so all three descriptions share identical abscissae with identical meaning: the
    lattice voxel weights the field at its node, so the comparable continuum mass is the window
    centred there, not the voxel-aligned window half a cell away.
    """
    per_seed = []
    for f in sorted(glob.glob(os.path.join(RDY, 'rf_s*.npz'))):
        z = np.load(f, allow_pickle=True)
        p = z['prof' + name][z['tau'] >= TAU_SS].mean(axis=0)
        p = np.roll(p, 1).reshape(-1, 2).sum(axis=1)
        per_seed.append(p / p.mean())
    return _fold(np.array(per_seed))


def lattice_profile(tag, name):
    """(phase, relative density, SEM over seeds) for one lattice arm, folded onto one period."""
    isp = SPECIES[name]
    per_seed = []
    for f in sorted(glob.glob(os.path.join(LAT, 'late_%s_s*.npz' % tag))):
        p = np.load(f, allow_pickle=True)['prof3'][isp, 0, :]
        per_seed.append((p / p.mean()).reshape(-1, NPER).mean(axis=0))
    return _fold(np.array(per_seed))


def _fold(per_seed):
    ph = np.arange(NPER) / float(NPER)
    sem = per_seed.std(axis=0, ddof=1) / np.sqrt(len(per_seed)) if len(per_seed) > 1 \
        else np.zeros(NPER)
    return ph, per_seed.mean(axis=0), sem


def counts(pats, root):
    """(tau, counts (frames, 3) normalized by N_A(0)) pooled over seeds, observation windows sorted.

    The three windows are separate runs from the same initial condition at different dt, so they
    are concatenated in tau rather than averaged; that is how the campaign was run.
    """
    tau, cnt = [], []
    for pat in pats:
        fs = sorted(glob.glob(os.path.join(root, pat)))
        if not fs:
            continue
        zs = [np.load(f, allow_pickle=True) for f in fs]
        tau.extend(zs[0]['tau'])
        cnt.extend(np.mean([z['counts'].astype(float) for z in zs], axis=0))
    o = np.argsort(tau)
    tau, cnt = np.array(tau)[o], np.array(cnt)[o]
    return tau, cnt / cnt[0, 0]


def counts_all():
    return {
        'ref': counts(('rf_vearly_s*.npz', 'rf_early_s*.npz', 'rf_s*.npz'), RDY),
        'exc': counts(('veryearly_excl_s*.npz', 'early_excl_s*.npz', 'late_excl_s*.npz'), LAT),
        'pt': counts(('veryearly_ideal_s*.npz', 'early_ideal_s*.npz', 'late_ideal_s*.npz'), LAT),
    }


def divergence(name, tag):
    """(tau, D_KL from the iPRD reference) for species `name` of arm `tag` in {excl, ideal}."""
    z = np.load(KL)
    return z['%s_%s_x' % (name, tag)], z['%s_%s_y' % (name, tag)]


def field_line(path=None):
    """sin(2 pi x / lambda) on the 32 lattice nodes -- the shape of Phi_s, without gamma."""
    path = path or os.path.join(LAT, 'late_excl_s0.npz')
    return np.load(path, allow_pickle=True)['psi0_line']


def slab_stack(path, tau_want, name='A', x0=0, nx=16):
    """(occupancy (nx, 1, 1), tau) for one run of voxels along the field axis at one frame.

    The value of a voxel here is the mean occupancy of the whole plane of voxels normal to the
    field that it belongs to, not its own count. That is forced by counting statistics: a voxel
    holds about three particles, so its Poisson fluctuation is 70 % of its mean against a field
    response of 17 %, and a single stack drawn from raw per-voxel counts shows nothing but shot
    noise. The plane averages 1024 of them and fluctuates by 2 %, so what the stack shows is the
    lattice's own occupancy profile at full resolution along the axis that matters.
    """
    z = np.load(path, allow_pickle=True)
    fi = int(np.argmin(np.abs(np.asarray(z['tau'], float) - tau_want)))
    occ = z['occ' + name][fi].reshape(N, N, N).astype(float)
    line = occ.mean(axis=(1, 2))[x0:x0 + nx]
    return line.reshape(nx, 1, 1), float(z['tau'][fi])


def field_axis_check(path=None):
    """Assert that axis 0 of the reshaped occupancy is the one carrying the ramp."""
    path = path or os.path.join(LAT, 'late_excl_s0.npz')
    z = np.load(path, allow_pickle=True)
    occ = z['occA'][-1].reshape(N, N, N).astype(float)
    amp = [np.ptp(occ.mean(axis=tuple(a for a in range(3) if a != ax))) / occ.mean()
           for ax in range(3)]
    assert amp[0] > 4 * max(amp[1], amp[2]), 'the ramp is not on axis 0: %r' % amp
    return amp
