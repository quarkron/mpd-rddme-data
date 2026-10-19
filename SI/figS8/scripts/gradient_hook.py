#!/usr/bin/env python3
"""The recording hook of gradient_growth.py: one frame of observables per call, and nothing else.

Split out of the driver on 2026-09-14 to keep every file under 500 lines. The driver owns the scene (the well, the
species, the registry, the reaction channels); this owns what a frame means. Per frame: the chemistry course, the
length histogram, the radial profiles of monomers, crowders, chain centres and summed chain length, each chain's
unwrapped centre of mass, the chain size, and -- for the field-free demixing runs -- the coarse-grained cross
correlation between polymer and crowder density, which is the observable a radial profile cannot express when the
structure has no centre.

Diagnostics behind environment switches: GRAD_BEAD_BODY (body centre to nearest subunit, every GRAD_BEAD_BODY_EVERY
frames), GRAD_TRACE_BODIES, GRAD_TRACE_INERT, GRAD_TRACE_NEWBORN, GRAD_DEMIX_CELL (the correlation's cell size in
voxels, default 4).
"""
import os
import time

import numpy as np
import gradient_bench as GB
import lm

import chain_geometry as G

H_NM = 10.0
M_ID, C_ID, R_ID = 1, 2, 2
FLAG_ACTIVE, FLAG_RIGID, FLAG_INERT, DIR_NONE, ROLE_BODY = 1, 0x4, 0x10, 255, 0


class Recorder(lm.DriftMpdRdmeSolver):
    def configure_recording(self, args, shells, n_shells, n_bodies=0):
        self.args = args; self.shells = shells; self.n_shells = n_shells; self.n_bodies = n_bodies
        self.times = []; self.n_chains = []; self.subunits = []; self.free_m = []; self.n_crowd = []
        self.len_hist = []; self.rad_m = []; self.rad_c = []; self.rad_com = []; self.rad_sub = []
        self.max_cid = 0; self.nucleated = []; self.last_backbones = None; self.rearmed = False
        self.com_rows = []                                  # (frame, cid, n, x, y, z) per chain per frame, x..z unwrapped voxels
        self.bound_fill = []; self.chain_voxels = []; self.last_overlay = None
        self.r2_ee = []; self.r2_g = []; self.n_sized = []      # chain size per frame (mean over chains): end-to-end and gyration
        self.rad_len_sum = []                                  # summed chain length per shell (with rad_com as the count: mean length vs radius)
        self.snaps = []; self.snap_at = set()                # filled by the driver from --snap-frames
        self.beads = []; self.save_beads = False             # --save-beads: (frames, chains, subunits, 3) voxel positions
        self.demix_crowd_var = []                            # crowder clustering per scale: the control the correlation needs
        self.demix_r = []; self.demix_contrast = []          # field-free demixing: one value per cell size, per frame
        self.demix_cells = []          # field-free demixing: polymer/crowder correlation per frame
        self.body_com = []                                    # --track-bodies: (frame, body, 3) RAW wrapped voxel centres
        self.bead_body_hist = np.zeros(160, np.int64)         # GRAD_BEAD_BODY: body centre to nearest chain subunit, 0.05-voxel bins
        self.frame_hash = []                                  # sha256 of the complex-id plane per frame (the structure loop's identity)
        self.t0 = time.perf_counter()

    def hookSimulation(self, t, lattice):
        if not self.rearmed:                                   # runSolver's init drops these: re-arm on the first hook
            novex = bool(getattr(self.args, 'no_vex', False))
            self.setStructureReciprocalVex(not novex, M_ID)
            if novex:                                          # --no-vex: the init would otherwise put both cores back
                self.setStructureBeadBodyExclusion(0); self.setStructureBodyPairExclusion(0)
            if novex or os.environ.get('GRAD_NO_BEAD_CORE'): self.setStructureBeadBodyExclusion(0)   # see configure()
            if not self.args.chemistry or self.args.n_monomers == 0: self.setStructureChainFriction(0)   # see configure()
            if self.args.subhop_q is not None: self.setStructureSubVoxelHop(int(self.args.subhop_q))
            if getattr(self.args, 'si_moves', False): self.setStructureInteriorMoveMask(0x1B)   # see configure()
            if getattr(self.args, 'interior_sweep', False): self.setStructureInteriorSweep(1)
            if getattr(self.args, 'chain_hop_scale', None) is not None: self.setStructureHopSpeciesScale(M_ID, self.args.chain_hop_scale)
            if os.environ.get('GRAD_ESCAPE_RATE'): self.setStructureHopEscapeRate(True)
            if self.args.crowder_mode == 'smear' and not novex and not os.environ.get('GRAD_NO_SMEAR'):
                self.setStructureSmearExtAll(1)
                if self.args.free_volume: self.setStructureFreeVolume(1)
            self.rearmed = True
        nx = self.args.nx
        if self.args.n_monomers > 0 or self.args.crowder_mode != 'smear':   # byte species exist: count them per voxel
            v = np.asarray(lattice.getParticleLatticeView())
            cm = (v == M_ID).sum(axis=(0, v.ndim - 1)).reshape(-1)  # (z, y, x) counts -> flat, same ordering as the well
            cc = (v == C_ID).sum(axis=(0, v.ndim - 1)).reshape(-1)
        else:                                                    # bodies and chains only: the lattice holds no particles, do not read it
            cm = np.zeros(nx ** 3); cc = np.zeros(nx ** 3)
        self.rad_m.append(np.bincount(self.shells, weights=cm, minlength=self.n_shells)[:self.n_shells])
        self._cc = np.bincount(self.shells, weights=cc, minlength=self.n_shells)[:self.n_shells]
        l1 = np.zeros(nx ** 3, np.int32); l2 = np.zeros(nx ** 3, np.int32); l3 = np.zeros(nx ** 3, np.int32)
        self.getStructureOverlay(l1, l2, l3)
        self.frame_hash.append(GB.digest(l1))
        # The topology commit re-numbers complexes by component label at every event, so a body is NOT "cid <= n_bodies":
        # read the registry flags (relabel-robust) and take the INERT bit as the body mark; chains are never inert.
        nreg = int(l1.max()) if l1.size else 0
        rf = np.zeros(max(nreg, 1), np.int32)
        if nreg > 0: self.getStructureRegistryFlags(rf)          # swig typemap: the length rides with the array
        is_body_cid = np.zeros(nreg + 1, bool); is_body_cid[1:] = (rf[:nreg] & FLAG_INERT) != 0
        self.body_cids = set(np.flatnonzero(is_body_cid).tolist())
        if os.environ.get('GRAD_TRACE_BODIES'):                                # debug: body centres vs their overlay voxel, drift toward the faces
            words = self.getStructureComplexPODSize() // 4; K = self.getStructureNComplexes()
            full = np.zeros(K * words, np.int32); self.getStructureRegistryFull(full)
            fl = full.reshape(K, words)[:, 16].astype(np.int64) & 0xFFFFFFFF; inert = (fl & FLAG_INERT) != 0
            com = np.frombuffer(full.tobytes(), np.float64).reshape(K, words // 2)[:, :3][inert]      # com_x,y,z = the first 3 doubles of the POD
            tipv = full.reshape(K, words)[:, 14][inert].astype(np.int64)                             # tip_neg = the overlay voxel
            vz, rem = np.divmod(tipv, nx * nx); vy, vx = np.divmod(rem, nx); vox = np.stack([vx, vy, vz], 1).astype(np.float64)
            c0 = (nx - 1) / 2.0; r_com = np.sqrt(((com - c0) ** 2).sum(1)); face_com = (np.minimum(com, nx - 1 - com).min(1) < 1.0).mean()
            try:                                                             # the hard core: the closest pair of body centres (voxels; sigma = 2.0006)
                from scipy.spatial import cKDTree
                d_pair, _ = cKDTree(com).query(com, k=2)
                dmin = float(d_pair[:, 1].min()); n_close = int((d_pair[:, 1] < 2.0006 * (1 - 1e-9)).sum())
                rr = r_com[d_pair[:, 1] < 2.0006 * (1 - 1e-9)]
                core_viol = int((rr < 5.0).sum())
            except Exception as e:
                dmin, n_close, core_viol = float('nan'), -1, -1
            print(f"[pairs] t={t:.3f} min centre distance {dmin:.4f} vox ({dmin*10:.2f} nm); bodies closer than sigma: {n_close} (of {inert.sum()}), of those at r<50 nm: {core_viol}", flush=True)
            print(f"[bodies] t={t:.3f} n={inert.sum()} <r_com>={r_com.mean():.1f} vox (uniform {np.sqrt(3*((nx*nx-1)/12.0)):.1f})  frac COM within 1 vox of a face {face_com:.3f} (uniform {1-(1-2/nx)**3:.3f})"
                  f"  max|COM-voxel| {np.abs(com - vox).max():.3f}  COM range [{com.min():.2f},{com.max():.2f}]  frac COM on sub-lattice {np.isclose(com*8, np.round(com*8)).all(1).mean():.3f}", flush=True)
        if os.environ.get('GRAD_TRACE_INERT'):                                 # debug: does any INERT complex have n != 1 (a chain wearing a body's flags)?
            words = self.getStructureComplexPODSize() // 4; K = self.getStructureNComplexes()
            full = np.zeros(K * words, np.int32); self.getStructureRegistryFull(full); full = full.reshape(K, words)
            nsub = full[:, 6].astype(np.int64); flags = full[:, 16].astype(np.int64) & 0xFFFFFFFF
            inert = (flags & FLAG_INERT) != 0; odd = inert & (nsub != 1)
            fills = np.bincount(l1[l1 > 0], minlength=K + 1)[1:K + 1]                 # voxels per cid from the overlay
            key = (int(inert.sum()), int(odd.sum()))
            if odd.any() and getattr(self, '_inert_dumps', 0) < 3 and hasattr(self, 'last_overlay'):   # anatomy of the first merged bodies
                self._inert_dumps = getattr(self, '_inert_dumps', 0) + 1
                p1, p2, p3 = self.last_overlay
                for i in np.flatnonzero(odd)[:3]:
                    vv = np.flatnonzero(l1 == i + 1); zz, rem = np.divmod(vv, nx * nx); yy, xx = np.divmod(rem, nx)
                    print(f"[inert-anatomy] t={t:.4f} cid{i+1} n={nsub[i]} flags={flags[i]:#x} tips=({full[i,13]},{full[i,14]}) com=({full[i,0:2]}) voxels: " +
                          " ".join(f"v{v}({x},{y},{z}):role{l3[v] >> 4}:fill{l3[v] & 0xF}:bond{l2[v]}|prev cid{p1[v]} role{p3[v] >> 4} fill{p3[v] & 0xF} bond{p2[v]}"
                                   for v, x, y, z in zip(vv, xx, yy, zz)), flush=True)
            if key != getattr(self, '_inert_key', None):
                self._inert_key = key
                sp = np.bincount((flags[inert] >> 8) & 0xFF, minlength=3)
                print(f"[inert] t={t:.4f} K={K} inert={key[0]} (n!=1: {key[1]}) species of inert {sp.tolist()} | chains n>1 non-inert "
                      f"{int(((~inert) & (nsub > 1)).sum())} | inert-with-n!=1 sample: " +
                      " ".join(f"cid{i+1}:n{nsub[i]}:vox{fills[i]}:f{flags[i]:#x}:tips({full[i,13]},{full[i,14]})" for i in np.flatnonzero(odd)[:6]), flush=True)
        chain_vox = (l1 > 0) & ~is_body_cid[l1]
        self.bound_fill.append(int((l3[chain_vox] & 0x0F).sum()))   # bound mass straight from the overlay (independent of the backbone walk)
        self.chain_voxels.append(int(chain_vox.sum()))
        self.last_overlay = (l1.copy(), l2.copy(), l3.copy())
        if os.environ.get('GRAD_TRACE_NEWBORN') and len(self.times) % 100 == 0:   # debug: monomer layer profile (face pile-up?)
            c3 = cm.reshape(nx, nx, nx)                                            # (z, y, x) as the view came out
            face = [c3[0].mean(), c3[-1].mean(), c3[:, 0].mean(), c3[:, -1].mean(), c3[:, :, 0].mean(), c3[:, :, -1].mean()]
            inner = c3[2:-2, 2:-2, 2:-2].mean(); pairs = (c3 >= 2)
            print(f"[trace] t={t:.5f} monomers/voxel faces z0 z1 y0 y1 x0 x1 = {np.round(face, 3).tolist()} interior {inner:.3f}; voxels with >=2: "
                  f"faces {int(pairs[0].sum() + pairs[-1].sum() + pairs[:, 0].sum() + pairs[:, -1].sum() + pairs[:, :, 0].sum() + pairs[:, :, -1].sum())} of {int(pairs.sum())}", flush=True)
        if os.environ.get('GRAD_TRACE_NEWBORN') and chain_vox.any():         # debug: every newborn voxel, every hook
            vv = np.flatnonzero(chain_vox); zz, rem = np.divmod(vv, nx * nx); yy, xx = np.divmod(rem, nx)
            print(f"[trace] t={t:.6f} step~{int(round(t / (self.args.tau_us * 1e-6)))} newborn voxels: " +
                  " ".join(f"cid{l1[v]}@({x},{y},{z}) fill{l3[v] & 0xF} role{l3[v] >> 4} bond{l2[v]}" for v, x, y, z in zip(vv, xx, yy, zz)), flush=True)
        # Bodies: radial histogram of their overlay voxels straight from the registry (tip_neg), vectorised; 12 500 bodies
        # walked one by one in Python cost 0.8 s per frame. Chains: the backbone walk over the chains' overlay only.
        c0 = (nx - 1) / 2.0
        body_rad = np.zeros(self.n_shells)
        if self.body_cids:
            words = self.getStructureComplexPODSize() // 4; K = self.getStructureNComplexes()
            full = np.zeros(K * words, np.int32); self.getStructureRegistryFull(full); full = full.reshape(K, words)
            inert = ((full[:, 16].astype(np.int64) & 0xFFFFFFFF) & FLAG_INERT) != 0
            tipv = full[:, 14][inert].astype(np.int64)
            vz, rem = np.divmod(tipv, nx * nx); vy, vx = np.divmod(rem, nx)
            rr = np.sqrt((vx - c0) ** 2 + (vy - c0) ** 2 + (vz - c0) ** 2).astype(int)
            body_rad = np.bincount(np.minimum(rr, self.n_shells - 1), minlength=self.n_shells)[:self.n_shells].astype(float)
            if getattr(self.args, 'track_bodies', False):
                # The bodies' centres, stored RAW and wrapped. Neither the unwrapping nor the identity of a body can be
                # settled here: the topology commit renumbers complexes by component label at every chemical event, so
                # row i of this frame is not necessarily the body that was row i of the last one. The analysis matches
                # frame to frame under the minimum image and reports how well separated that match was.
                com = np.frombuffer(full.tobytes(), np.float64).reshape(K, words // 2)[:, :3][inert]
                self.body_com.append(com.astype(np.float32))
        l1c = np.where(is_body_cid[l1], 0, l1)
        # DEMIXING OBSERVABLE (field-free runs): a radial profile cannot express a structure with no centre, so coarse
        # grain the box and correlate polymer density with crowder density. Excluded volume alone makes the two
        # anti-correlate at the PACKING scale even when perfectly mixed, so the measurement is made at several cell
        # sizes: demixing is growth of the anti-correlation at the LARGE ones, where packing says nothing.
        cells = [int(c) for c in os.environ.get('GRAD_DEMIX_CELLS', '2 4 8 16').split() if nx % int(c) == 0]
        if cells:
            polv = (l1c > 0).reshape(nx, nx, nx).astype(np.float64)
            crwv = (cc if self.n_bodies == 0 else is_body_cid[l1].astype(np.float64)).reshape(nx, nx, nx)
            rs, ks = [], []
            for cell in cells:
                nc = nx // cell
                blk = lambda f: f.reshape(nc, cell, nc, cell, nc, cell).sum(axis=(1, 3, 5)).ravel()
                pol, crw = blk(polv), blk(crwv)
                if pol.std() > 0 and crw.std() > 0:
                    rs.append(float(np.corrcoef(pol, crw)[0, 1]))
                    q = np.argsort(pol); k = max(1, len(pol) // 10)
                    lo = crw[q[:k]].mean(); hi = crw[q[-k:]].mean()      # crowders where polymer is scarcest / densest
                    ks.append(float(hi / lo) if lo > 0 else float('nan'))
                else:
                    rs.append(float('nan')); ks.append(float('nan'))
            self.demix_cells = cells
            self.demix_r.append(rs); self.demix_contrast.append(ks)
            # The correlation above cannot on its own distinguish demixing from GAP FILLING: the polymer is excluded
            # from the crowders' hard cores and from nucleating inside them, so it fills whatever space is left and
            # anti-correlates with them however they are arranged. What separates the two is whether the CROWDERS
            # themselves cluster. Their normalised count variance per cell (1 for an ideal gas, below 1 for a
            # hard-sphere fluid, rising if they aggregate) is the observable that answers it, and the run's own early
            # frames -- crowders freshly placed, no polymer yet -- are its baseline.
            self.demix_crowd_var.append([float(np.var(blkc) / np.mean(blkc)) if np.mean(blkc) > 0 else float('nan')
                                         for blkc in (crwv.reshape(nx // c, c, nx // c, c, nx // c, c).sum(axis=(1, 3, 5)).ravel()
                                                      for c in cells)])
        if self.snap_at and len(self.times) in self.snap_at:      # spatial snapshots for the figure
            self.snaps.append((float(t), np.flatnonzero(is_body_cid[l1]).astype(np.int32),
                               np.flatnonzero(l1c > 0).astype(np.int32),
                               cc.astype(np.int16) if self.n_bodies == 0 else None))   # point mode: the byte bath per voxel
        chains = G.backbones(l1c, l2, nx, nx, nx)
        if self.save_beads:                                      # per-frame backbones: what a Rouse-mode transform needs
            ok = [np.asarray(ch['pos'], np.int32) for c, ch in sorted(chains.items()) if ch.get('ok')]
            if ok and all(len(q) == len(ok[0]) for q in ok):
                self.beads.append(np.stack(ok))
        lens = []; coms = []; subs = np.zeros(self.n_shells); ree = []; rg = []
        for cid, ch in chains.items():
            if not ch.get('ok'): continue
            pos = np.asarray(ch['pos'], np.float64)           # (n, 3) voxel coordinates, unwrapped along the backbone
            lens.append(len(pos))
            if len(pos) > 1:                                  # chain size, for the crowder-compaction ladder (Fig 3)
                ree.append(float(((pos[-1] - pos[0]) ** 2).sum()))
                rg.append(float(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))
            com_raw = pos.mean(axis=0)
            com = com_raw % nx
            coms.append(com)
            self.com_rows.append((len(self.times), int(cid), len(pos), com_raw[0], com_raw[1], com_raw[2]))
            for p in pos:
                q = np.asarray(p) % nx
                vi = int(q[2]) * nx * nx + int(q[1]) * nx + int(q[0])
                subs[self.shells[vi]] += 1.0
            self.max_cid = max(self.max_cid, int(cid))
        self.times.append(float(t)); self.n_chains.append(len(lens)); self.subunits.append(int(sum(lens)))
        self.free_m.append(int(cm.sum())); self.n_crowd.append(int(body_rad.sum()) if self.n_bodies > 0 else int(cc.sum()))
        self.len_hist.append(np.bincount(np.asarray(lens, np.int64), minlength=nx + 1)[:nx + 1] if lens else np.zeros(nx + 1, np.int64))
        rc = np.zeros(self.n_shells)
        lsum = np.zeros(self.n_shells)                       # chain LENGTH by the shell its centre sits in, so the
        for com, ln in zip(coms, lens):                      # steady-state length can be read against radius
            c = (nx - 1) / 2.0
            rr = np.sqrt(((com - c) ** 2).sum())
            sh = min(int(rr), self.n_shells - 1)
            rc[sh] += 1.0; lsum[sh] += float(ln)
        self.rad_len_sum.append(lsum)
        self.r2_ee.append(float(np.mean(ree)) if ree else 0.0); self.r2_g.append(float(np.mean(rg)) if rg else 0.0)
        self.n_sized.append(len(ree))
        self.rad_c.append(body_rad if self.n_bodies > 0 else self._cc)   # smear: bodies per shell; point: byte crowders per shell
        self.rad_com.append(rc); self.rad_sub.append(subs); self.nucleated.append(len(lens))   # relabelling makes 'max cid' meaningless: record the live chain count
        self.last_backbones = {int(cid): np.asarray(ch['pos'], np.int64) for cid, ch in chains.items() if ch.get('ok')}
        if len(self.times) % 50 == 1:
            print(f"[gradient] t={t:.3f}s chains={len(lens)} subunits={sum(lens)} free M={int(cm.sum())} bodies={len(self.body_cids)} "
                  f"wall={time.perf_counter() - self.t0:.0f}s", flush=True)
        return 0
