"""Joint (n_A, n_B) histogram of the ReaDDy ternary reference at tau = 0 over the FILLED half, for seeding the lattice.

The ReaDDy arm starts A and B by random sequential insertion into x < L/2, which cannot pile spheres up, so its
filled half is already correlated within and between species. A multinomial lattice start would begin in a state the
reference can never occupy. ternary_relax.py --seed-hist reproduces P(n_A, n_B, 0) by drawing per-voxel pairs from
this histogram, so any later separation between the descriptions is dynamics (the same device as the binary figure,
vex_relax2.py / bd2_seed_joint.npy).

  python make_seed_hist.py out_three_relax/rt3_s*.npz -o out_three_relax/rt3_seed_joint.npy
"""
import argparse, glob, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('npz', nargs='+')
ap.add_argument('-o', '--out', required=True)
A = ap.parse_args()
files = sorted(f for pat in A.npz for f in glob.glob(pat))
if not files:
    sys.exit('no input files')
J = None; used = 0; nmax_seen = 0
for f in files:
    z = np.load(f, allow_pickle=True); m = json.loads(str(z['meta']))
    mcell = int(round(m['L'] / m['h'])); frac = m['frac']
    a0 = z['occA'][0].reshape(mcell, mcell, mcell); b0 = z['occB'][0].reshape(mcell, mcell, mcell)
    filled = np.arange(mcell) < int(round(frac * mcell))          # x is the slowest index of the ReaDDy binning
    a = a0[filled].ravel(); b = b0[filled].ravel()
    nmax_seen = max(nmax_seen, int((a + b).max()))
    h = np.zeros((int(a.max()) + 1, int(b.max()) + 1))
    np.add.at(h, (a, b), 1)
    if J is None:
        J = h
    else:
        H = np.zeros((max(J.shape[0], h.shape[0]), max(J.shape[1], h.shape[1])))
        H[:J.shape[0], :J.shape[1]] += J; H[:h.shape[0], :h.shape[1]] += h; J = H
    used += 1
np.save(A.out, J)
tot = J.sum(); nA = (J * np.arange(J.shape[0])[:, None]).sum() / tot; nB = (J * np.arange(J.shape[1])[None, :]).sum() / tot
print(f"{used} seeds, {int(tot)} filled voxels: <n_A> {nA:.2f}, <n_B> {nB:.2f}, max n_A+n_B {nmax_seen} (lattice cap 64) -> {A.out}")
