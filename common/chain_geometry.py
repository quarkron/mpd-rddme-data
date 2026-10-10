#!/usr/bin/env python3
"""Chain geometry read back from the structure overlay.

The engine stores a filament as an occupied-voxel field plus a per-voxel bond direction byte that
points at the next subunit toward the plus tip. Walking that successor chain from the minus tip
recovers the ordered backbone, which is everything the conformational observables need. The engine
exposes no geometry of its own; existing analysis walks the same chain and then reduces it to a
subunit count, discarding exactly the ordering these observables require.

Positions are absolute lattice coordinates of the minus tip, then unwrapped along the walk, so a
chain crossing a periodic face still yields a correct end to end vector. Reducing positions modulo
the box would make a crossing chain look folded and corrupt every length-like observable. Keeping
them absolute is what makes displacement between frames meaningful; a walk that starts at the origin
each frame gives correct shapes and destroys every displacement.

Between frames the chain itself can cross a face, so a displacement taken directly from two frames
can jump by a box length. Callers measuring displacement must unwrap frame to frame against the
previous position, which the dynamics campaign does.

All lengths are in voxels. Multiply by the voxel edge for physical units.

The L26 direction index ordering here must match the engine's byte exactly. It is x fastest, then y,
then z, with the zero vector skipped at the midpoint. Enumerating the same 26 vectors in any other
order gives a module that passes every self-consistent test and produces wrong geometry against real
engine output.
"""
import numpy as np

DIR_NONE = 255


def l26_vec(idx):
    """The engine's L26 direction index, x fastest then y then z, with the zero vector skipped.

    This ordering is a wire format, not a convention this module is free to choose. A module that
    enumerates the same 26 vectors in a different order is self-consistent and silently wrong the
    moment it reads a bond byte the engine wrote.
    """
    raw = idx if idx < 13 else idx + 1
    return ((raw % 3) - 1, ((raw // 3) % 3) - 1, (raw // 9) - 1)


def l26_index(dx, dy, dz):
    raw = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
    return raw if raw < 13 else raw - 1


VECS = np.array([l26_vec(i) for i in range(26)], np.int64)
LENS = np.linalg.norm(VECS, axis=1)
UNITS = VECS / LENS[:, None]
COSTHETA = np.clip(UNITS @ UNITS.T, -1.0, 1.0)


def _offsets(lx, ly):
    return VECS[:, 0] + lx * VECS[:, 1] + lx * ly * VECS[:, 2]


def backbones(l1, l2, lx, ly, lz):
    """Ordered backbones of every complex in the overlay.

    Returns a dict from complex identifier to a record holding the voxel index sequence, the bond
    direction sequence, and the unwrapped integer positions. A complex whose successor walk does not
    terminate, which would mean a cycle or a corrupt bond field, is returned with a flag rather than
    silently truncated, because a truncated chain reports a plausible and wrong end to end distance.
    """
    l1 = np.asarray(l1).ravel()
    l2 = np.asarray(l2).ravel()
    off = _offsets(lx, ly)
    xyz = lx * ly * lz

    has_bond = (l2 >= 0) & (l2 < 26) & (l1 > 0)
    succ = {}
    for v in np.flatnonzero(has_bond).tolist():
        succ[v] = v + int(off[int(l2[v])])
    pointed = set(succ.values())

    out = {}
    for cid in np.unique(l1[l1 > 0]).tolist():
        members = set(np.flatnonzero(l1 == cid).tolist())
        starts = [v for v in members if v not in pointed]
        if not starts:
            out[int(cid)] = {'ok': False, 'reason': 'no minus tip, chain is cyclic or corrupt'}
            continue
        v = starts[0]
        vox, dirs, pos = [], [], []
        # Absolute lattice coordinates of the minus tip, then unwrapped along the walk. Starting the
        # walk at the origin instead would give correct shape observables and destroy every
        # displacement observable, because the chain's position would be defined relative to itself.
        cur = np.array([v % lx, (v // lx) % ly, v // (lx * ly)], np.int64)
        seen = set()
        ok = True
        while True:
            if v in seen or len(vox) > xyz:
                ok = False
                break
            seen.add(v)
            vox.append(v)
            pos.append(cur.copy())
            d = int(l2[v]) if 0 <= int(l2[v]) < 26 else -1
            if d < 0:
                break
            nxt = v + int(off[d])
            if nxt not in members:
                ok = False
                break
            dirs.append(d)
            cur = cur + VECS[d]
            v = nxt
        out[int(cid)] = {'ok': ok, 'voxels': np.array(vox, np.int64),
                         'dirs': np.array(dirs, np.int64),
                         'pos': np.array(pos, np.int64) if pos else np.zeros((0, 3), np.int64),
                         'n': len(vox)}
    return out


def observables(chain):
    """Conformational observables of one ordered backbone.

    The radius of gyration is taken about the subunit centroid. The end to end vector is between the
    two terminal subunit centres, so it is one bond shorter than the contour, which is the convention
    the transfer matrix reference uses as well.
    """
    pos = chain['pos'].astype(float)
    dirs = chain['dirs']
    n = len(pos)
    if n < 2:
        return {'n': n, 'r_ee': 0.0, 'r_ee2': 0.0, 'rg2': 0.0, 'contour': 0.0,
                'mean_cos': float('nan'), 'n_joints': 0}
    ree = pos[-1] - pos[0]
    centre = pos.mean(axis=0)
    rg2 = float(((pos - centre) ** 2).sum(axis=1).mean())
    contour = float(LENS[dirs].sum())
    if len(dirs) >= 2:
        cos = COSTHETA[dirs[:-1], dirs[1:]]
    else:
        cos = np.zeros(0)
    return {'n': n, 'r_ee': float(np.linalg.norm(ree)), 'r_ee2': float(ree @ ree),
            'rg2': rg2, 'contour': contour, 'n_bonds': len(dirs),
            'mean_cos': float(cos.mean()) if cos.size else float('nan'),
            'n_joints': int(cos.size), 'bond_lengths': LENS[dirs]}


def bond_correlation(chain, max_sep):
    """Bond vector correlation against contour separation, for one configuration.

    Averaged over all pairs at each separation. A single configuration gives a very noisy estimate,
    so this is meant to be accumulated across frames rather than read one frame at a time.
    """
    dirs = chain['dirs']
    b = VECS[dirs].astype(float)
    m = len(b)
    out = np.full(max_sep + 1, np.nan)
    for s in range(0, min(max_sep, m - 1) + 1):
        if m - s <= 0:
            break
        out[s] = float((b[:m - s] * b[s:]).sum(axis=1).mean())
    return out


def rouse_modes(chain, p_max):
    """Rouse mode amplitudes of one configuration.

    Mode p is the discrete cosine projection of the subunit positions, which is the standard
    definition for a chain with free ends. Mode zero is the centre of mass and is returned so a
    caller can separate centre of mass motion from internal motion without recomputing it.
    """
    pos = chain['pos'].astype(float)
    n = len(pos)
    idx = np.arange(n)
    modes = np.empty((p_max + 1, 3))
    for p in range(p_max + 1):
        w = np.cos(p * np.pi * (idx + 0.5) / n)
        modes[p] = (w[:, None] * pos).sum(axis=0) / n
    return modes


def read_chains(sol, lx, ly, lz):
    """Read the overlay from a solver and return the ordered backbones."""
    xyz = lx * ly * lz
    l1 = np.zeros(xyz, np.int32)
    l2 = np.zeros(xyz, np.int32)
    l3 = np.zeros(xyz, np.int32)
    sol.getStructureOverlay(l1, l2, l3)
    return backbones(l1, l2, lx, ly, lz)
