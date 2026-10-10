"""Ribosome as a rigid multi-voxel blob (add-nucleoid-vex-partitioning-force,
capability `ribosome-crowder-bodies`, tasks 3.1 + 3.2).

A 20 nm ribosome is one connected ~33-voxel rigid sphere (radius 2 voxels at
5 nm/voxel) seeded as a SINGLE mesh-mode structure complex carrying
`STRUCT_FLAG_RIGID`: it translates via the collective rigid hop and reorients via
the blob rotation, but performs no interior conformational moves. A subunit stays
one voxel — the 20 nm span is the footprint, not σ (sidestepping the White-Bear
σ ≤ voxel-edge constraint).

The two entry points:
  * `blob_template()`   — the fixed sphere offsets + intra-blob adjacency edges,
                          computed once (task 3.1).
  * `seed_blobs()`      — vectorized numpy scatter of N blobs at given centres into
                          the overlay / mesh / registry arrays, with NO per-bead
                          Python `addParticleAtIdx` loop (task 3.2).

Wire the returned arrays into a `DriftMpdRdmeSolver` with:
    sol.setStructureMeshMode(1)
    sol.setStructureOverlay(l1, l2, l3)
    sol.setStructureMeshOverlay(adj, plus)
    sol.setStructureRegistry(n, sizes, coms, flags)
    for i in range(n): sol.setStructureTips(i, tips[i], tips[i], -1)
    sol.runStructureTopologyPass()          # classifies each blob MESH, sets rep/pivot
"""
import numpy as np

# --- Smeared fundamental-measure (FMT) volume exclusion ---------------------------------------------
# A ribosome deposits its TRUE hard-sphere Rosenfeld scalar weighted densities (n0..n3) over the fixed
# 3^3 voxel stencil, so its excluded volume is faithful to the real sphere (correct n3 volume AND n2
# surface, not a per-voxel-sigma dump). `fmt_stencil` builds the (27,4) stencil; `deposit_smear_field`
# sums it over all ribosome centres into the per-voxel field the White-Bear kernel folds in. The kernel
# subtracts each body's own (frozen) smear per move ⇒ no spurious self-force. Ordering matches the device:
# offset index = ((dx+1)*3+(dy+1))*3+(dz+1), dx,dy,dz in -1..1; field layout [alpha*XYZ + v].
_STENCIL_OFFS = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                         np.int64)  # index k == ((dx+1)*3+(dy+1))*3+(dz+1)


def stencil_radius(diam_nm, voxel_nm):
    """Support radius R (voxels) so that every voxel a sphere of `diam_nm` centred on a voxel intersects lies within
    |offset| <= R: R = floor(r/h + 0.5). 20 nm on 10 nm -> 1 (3^3); on 5 nm -> 2 (5^3); on 3.33 nm -> 3 (7^3)."""
    return max(1, int(np.floor(0.5 * diam_nm / voxel_nm + 0.5)))


def stencil_offsets(R):
    """((2R+1)^3, 3) integer offsets in the kernel's order (x slowest): row k == offset of stencil row k."""
    r = range(-R, R + 1)
    return np.array([[dx, dy, dz] for dx in r for dy in r for dz in r], np.int64)


def fmt_stencil(diam_nm=20.0, voxel_nm=10.0, radius=None):
    """((2R+1)^3, 4) FMT scalar weighted-density stencil [n0,n1,n2,n3] per voxel offset for a hard sphere of `diam_nm`
    centred on a voxel; R = `radius` or stencil_radius(). n3 = volume-overlap fraction, n2 = surface-area/Vvox,
    n1 = n2/(4πR), n0 = n2/(4πR²). Deterministic. Σn3·V = (π/6)d³, Σn2·V = πd². R = 1 is the original 27-row stencil,
    row k == offset stencil_offsets(1)[k] (x slowest), which is the kernel's [((dx+R)(2R+1)+(dy+R))(2R+1)+(dz+R)]."""
    R = stencil_radius(diam_nm, voxel_nm) if radius is None else int(radius)
    W = 2 * R + 1
    Rs = diam_nm / 2.0; V = voxel_nm ** 3
    m = 64 if R == 1 else max(16, 64 // R)                     # sub-samples per voxel edge; keep the grid ~<= 200^3
    ax = np.arange(-(R + 0.5) * voxel_nm, (R + 0.5) * voxel_nm, voxel_nm / m) + voxel_nm / (2 * m)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij"); dv = (voxel_nm / m) ** 3
    insph = (X * X + Y * Y + Z * Z) <= Rs * Rs
    ix = np.round(X / voxel_nm).astype(int); iy = np.round(Y / voxel_nm).astype(int); iz = np.round(Z / voxel_nm).astype(int)
    key = ((ix + R) * W + (iy + R)) * W + (iz + R)
    vol = np.bincount(key.ravel()[insph.ravel()], minlength=W ** 3).astype(float) * dv
    # surface via a deterministic Fibonacci sphere
    Ns = 400000
    i = np.arange(Ns) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / Ns); th = np.pi * (1.0 + 5.0 ** 0.5) * i
    px = Rs * np.cos(th) * np.sin(phi); py = Rs * np.sin(th) * np.sin(phi); pz = Rs * np.cos(phi)
    jx = np.round(px / voxel_nm).astype(int); jy = np.round(py / voxel_nm).astype(int); jz = np.round(pz / voxel_nm).astype(int)
    inb = (np.abs(jx) <= R) & (np.abs(jy) <= R) & (np.abs(jz) <= R)
    skey = (((jx + R) * W + (jy + R)) * W + (jz + R))[inb]
    area = np.bincount(skey, minlength=W ** 3).astype(float) / Ns * (4.0 * np.pi * Rs * Rs)
    n3 = vol / V; n2 = area / V; n1 = n2 / (4.0 * np.pi * Rs); n0 = n2 / (4.0 * np.pi * Rs * Rs)
    return np.stack([n0, n1, n2, n3], 1)                     # ((2R+1)^3, 4), row k == offset stencil_offsets(R)[k]


def deposit_smear_field(centers, LX, LY, LZ, stencil):
    """Per-voxel weighted-density field (float64, len 4*LX*LY*LZ, layout [alpha*XYZ + v]) = sum over
    `centers` of `stencil` placed at each centre. Also returns the frozen-centre array (int32, 3*N,
    layout [3*cid + k]) for the kernel's per-move self-subtraction. Fully vectorized."""
    centers = np.asarray(centers, np.int64); N = centers.shape[0]; XYZ = LX * LY * LZ
    xi = np.zeros(4 * XYZ, np.float64)
    if N:
        R = int(round((stencil.shape[0] ** (1.0 / 3.0) - 1) / 2)); offs = stencil_offsets(R)
        vox = centers[:, None, :] + offs[None, :, :]                # (N,(2R+1)^3,3)
        inb = ((vox[..., 0] >= 0) & (vox[..., 0] < LX) & (vox[..., 1] >= 0) & (vox[..., 1] < LY)
               & (vox[..., 2] >= 0) & (vox[..., 2] < LZ))
        v = vox[..., 2] * (LX * LY) + vox[..., 1] * LX + vox[..., 0]
        for a in range(4):
            vals = np.broadcast_to(stencil[:, a], (N, offs.shape[0]))[inb]
            np.add.at(xi, a * XYZ + v[inb], vals)
    frozen = centers.astype(np.int32).reshape(-1)                # [3*cid + k]
    return xi, frozen


DIR_NONE  = 255
ROLE_BODY = 0
# Registry flag bits (mirror struct_complex_pod.h).
STRUCT_FLAG_ACTIVE   = 0x1
STRUCT_FLAG_RIGID    = 0x4
STRUCT_FLAG_CONFINED = 0x8
STRUCT_MAX_SUB       = 64            # per-complex gather / collective-move cap
BLOB_RADIUS_VOX      = 2.0          # 20 nm diameter at 5 nm/voxel


# --- L26 direction packing (identical convention to the kernel / mesh tests) -------------
def _l26(dx, dy, dz):
    """Pack a unit-ish step (each component in {-1,0,1}, not all zero) into 0..25."""
    raw = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
    return raw if raw < 13 else raw - 1          # 13 == (0,0,0) self, skipped


def _opp(d):
    raw = d if d < 13 else d + 1
    return _l26(-((raw % 3) - 1), -((raw // 3) % 3 - 1), -((raw // 9) - 1))


def blob_offsets(radius_vox=BLOB_RADIUS_VOX):
    """Integer voxel offsets (dx,dy,dz) of a solid sphere of `radius_vox`, origin-centred,
    with (0,0,0) FIRST (used as the gather-start / tip voxel). radius 2 -> 33 voxels."""
    r = int(np.floor(radius_vox)); r2 = radius_vox * radius_vox + 1e-9
    offs = [(0, 0, 0)]
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if (dx, dy, dz) != (0, 0, 0) and dx * dx + dy * dy + dz * dz <= r2:
                    offs.append((dx, dy, dz))
    return np.array(offs, np.int64)


def _intra_edges(offsets):
    """Every unordered pair of blob voxels that are 26-neighbours (Chebyshev dist 1) ->
    one internal bond. Returns (i, j, dij, dji) with i<j (local indices) and the packed
    L26 direction each way. O(nsub^2) but computed ONCE for the template."""
    n = len(offsets)
    ii, jj, dij, dji = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = offsets[j] - offsets[i]
            if int(np.max(np.abs(d))) == 1:                 # 26-neighbour -> bonded
                ii.append(i); jj.append(j)
                dij.append(_l26(int(d[0]), int(d[1]), int(d[2])))
                dji.append(_opp(dij[-1]))
    return (np.array(ii, np.int64), np.array(jj, np.int64),
            np.array(dij, np.int64), np.array(dji, np.int64))


def blob_template(radius_vox=BLOB_RADIUS_VOX):
    """(offsets, edges) for a rigid ribosome blob. `edges` = (i, j, dij, dji). Rejects a
    blob that would exceed STRUCT_MAX_SUB (loud — the collective move cannot gather it)."""
    offsets = blob_offsets(radius_vox)
    if len(offsets) > STRUCT_MAX_SUB:
        raise ValueError(
            f"ribosome blob of radius {radius_vox} voxels = {len(offsets)} voxels "
            f"exceeds STRUCT_MAX_SUB={STRUCT_MAX_SUB}; the collective rigid move cannot "
            f"gather it as one unit. Shrink the radius or raise STRUCT_MAX_SUB.")
    return offsets, _intra_edges(offsets)


def seed_blobs(centers, LX, LY, LZ, radius_vox=BLOB_RADIUS_VOX,
               species_tag=0, confined=True, first_cid=1, template=None):
    """Vectorized seed of `centers.shape[0]` rigid blobs (task 3.2).

    centers : (N,3) int voxel coords of each blob centre. Callers must guarantee the
              footprints are disjoint and in-lattice (see `pack_centers`).
    Returns a dict of arrays sized for the solver setters:
      l1, l2, l3           overlay (complex-id / bond byte=NONE / role|fill)
      adj, plus            mesh adjacency + plusward bitmasks
      sizes, coms, flags   registry (subunit count, COM, flag bitmask)
      tips                 (N,) gather-start voxel (the centre) per blob

    Exact mass: every blob contributes len(offsets) voxels; no per-bead Python loop
    (a single numpy scatter over N*nsub voxels + N*nedges bonds)."""
    centers = np.asarray(centers, np.int64)
    N = centers.shape[0]
    offsets, (ei, ej, edij, edji) = template if template is not None else blob_template(radius_vox)
    nsub = len(offsets)
    XYZ = LX * LY * LZ

    l1  = np.zeros(XYZ, np.int32)
    l2  = np.full(XYZ, DIR_NONE, np.int32)
    l3  = np.zeros(XYZ, np.int32)
    adj = np.zeros(XYZ, np.int32)
    plus = np.zeros(XYZ, np.int32)
    if N == 0:
        return dict(l1=l1, l2=l2, l3=l3, adj=adj, plus=plus,
                    sizes=np.zeros(0, np.int32), coms=np.zeros(0, np.float64),
                    flags=np.zeros(0, np.int32), tips=np.zeros(0, np.int64), nsub=nsub)

    # (N, nsub, 3) absolute voxel coords, then flat global indices (N, nsub).
    vox = centers[:, None, :] + offsets[None, :, :]
    if vox.min() < 0 or (vox[..., 0].max() >= LX or vox[..., 1].max() >= LY or vox[..., 2].max() >= LZ):
        raise ValueError("a ribosome blob footprint falls outside the lattice; "
                         "keep centres >= radius+1 from every wall.")
    gidx = (vox[..., 2] * (LX * LY) + vox[..., 1] * LX + vox[..., 0]).astype(np.int64)  # (N,nsub)

    cids = first_cid + np.arange(N, dtype=np.int32)          # 1-based complex ids
    flat = gidx.reshape(-1)
    if np.unique(flat).size != flat.size:
        raise ValueError("overlapping ribosome blob footprints (a voxel is claimed by "
                         "two blobs); increase the centre spacing.")
    l1[flat] = np.repeat(cids, nsub)
    l3[flat] = (ROLE_BODY << 4) | 1                          # every voxel: body, fill 1

    # Intra-blob bonds, vectorized over all N blobs. bitwise_or.at accumulates the
    # multiple direction bits a voxel receives from its several neighbours.
    src = gidx[:, ei].reshape(-1)                            # (N*E,)
    dst = gidx[:, ej].reshape(-1)
    bit_ij = (np.int32(1) << edij.astype(np.int32))          # (E,)
    bit_ji = (np.int32(1) << edji.astype(np.int32))
    tij = np.tile(bit_ij, N); tji = np.tile(bit_ji, N)
    np.bitwise_or.at(adj, src, tij)
    np.bitwise_or.at(adj, dst, tji)
    np.bitwise_or.at(plus, src, tij)                         # plusward on the lower-local-index end only

    flag = STRUCT_FLAG_ACTIVE | STRUCT_FLAG_RIGID | (int(species_tag) << 8)
    if confined:
        flag |= STRUCT_FLAG_CONFINED
    sizes = np.full(N, nsub, np.int32)
    coms  = centers.astype(np.float64).reshape(-1)           # sphere COM == centre; flat (3N,)
    flags = np.full(N, flag, np.int32)
    tips  = gidx[:, 0].copy()                                # offsets[0] == (0,0,0) centre
    return dict(l1=l1, l2=l2, l3=l3, adj=adj, plus=plus,
                sizes=sizes, coms=coms, flags=flags, tips=tips, nsub=nsub)


def pack_centers(n, center, radius_region_vox, rng, min_sep=None, radius_vox=BLOB_RADIUS_VOX,
                 max_tries_factor=200):
    """Random sequential placement of `n` blob centres inside a sphere of
    `radius_region_vox` about `center`, no two closer than `min_sep` (default 5 voxels,
    which guarantees disjoint radius-2 footprints: |Δ|>=5 > 2+2). Uses a coarse spatial
    hash so the neighbour test is O(1); vectorized candidate batches. Raises loudly if it
    cannot fit `n` (no silent truncation)."""
    if min_sep is None:
        min_sep = 2.0 * radius_vox + 1.0                     # 5 voxels for radius 2
    cx, cy, cz = center
    lo = radius_vox + 1.0                                    # keep the footprint off the wall
    rmax = radius_region_vox - lo
    if rmax <= 0:
        raise ValueError("region too small for even one blob")
    cell = float(min_sep)
    grid = {}                                                # (i,j,k) cell -> list of centres
    accepted = []
    tries = 0; max_tries = int(max_tries_factor * n) + 1000
    ms2 = min_sep * min_sep
    while len(accepted) < n and tries < max_tries:
        # a batch of uniform-in-ball candidates (vectorized)
        m = max(64, (n - len(accepted)))
        u = rng.random(m); rad = rmax * np.cbrt(u)
        dirv = rng.normal(size=(m, 3)); dirv /= np.linalg.norm(dirv, axis=1, keepdims=True)
        cand = np.array([cx, cy, cz]) + rad[:, None] * dirv
        cand = np.round(cand).astype(np.int64)
        for p in cand:
            tries += 1
            ci, cj, ck = int(p[0] // cell), int(p[1] // cell), int(p[2] // cell)
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        for q in grid.get((ci + di, cj + dj, ck + dk), ()):
                            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < ms2:
                                ok = False; break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if ok:
                accepted.append(p); grid.setdefault((ci, cj, ck), []).append(p)
                if len(accepted) == n: break
    if len(accepted) < n:
        raise RuntimeError(
            f"could only place {len(accepted)}/{n} ribosome blobs (min_sep={min_sep} vox) "
            f"in a sphere of radius {radius_region_vox:.1f} vox after {tries} tries — the "
            f"packing fraction is too high; reduce n or the blob radius.")
    return np.array(accepted, np.int64)
