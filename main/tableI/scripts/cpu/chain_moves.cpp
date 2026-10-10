// chain_moves.cpp -- the SI move set (LM-RDDME SI, sec:fcc-moveset): the chain's collective hop by a
// face vector (mpd_structure_hop_kernel, whole-voxel Metropolis path) and the interior moves
// (mpd_structure_interior_kernel with mask 0x1B: end-bond rotation, full kink = flip + sideways
// (struct_full_kink_dev.cuh), edge translation of a segment (struct_edge_translation*_dev.cuh)),
// every complex once per step. No crankshaft and no rigid rotation: the SI set has neither.
//
// Every move: geometry and occupancy first, then the chain friction thinning, then the bead-body hard
// core, then the move energy (bend and stretch where the move changes them, plus the VEX terms the
// device evaluates), then Metropolis. The device claims voxels with atomicCAS against concurrent
// movers; one core moves one complex at a time, so the claims are dropped.
#include "chain_common.h"
#include "structure.h"
#include <cmath>
#include <cstdlib>
#include <vector>

static inline bool isChain(const Complex& c) { return c.active && !c.body; }

// Move complex k's subunits to dest[] (same path order) and rewrite its overlay from the path.
static void relocate(Sim& s, int k, const std::vector<int>& dest) {
    Complex& c = s.cx[k];
    const uint32_t cid = (uint32_t)(k + 1);
    for (int v : c.vox) if (s.l1[v] == cid) clearVoxel(s, v);
    c.vox = dest;
    writeChainOverlay(s, k);
}

static double sumVex(const Sim& s, int cid, int species, const int* src, const int* dst, int n) {
    double d = 0.0;
    for (int i = 0; i < n; ++i) d += vexInsertDelta(s, cid, species, src[i], dst[i]);
    return d;
}

static inline bool metropolis(Sim& s, double dU, float u) {
    const float acc = (dU <= 0.0) ? 1.0f : expf(-(float)dU);
    return !(acc < 1.0f && u >= acc);
}

void chainHopAll(Sim& s) {
    static const int FX[6] = {1, -1, 0, 0, 0, 0}, FY[6] = {0, 0, 1, -1, 0, 0}, FZ[6] = {0, 0, 0, 0, 1, -1};
    std::vector<int> dest;
    for (size_t k = 0; k < s.cx.size(); ++k) {
        Complex& c = s.cx[k];
        if (!isChain(c)) continue;
        const int n = (int)c.vox.size(), cid = (int)k + 1;
        float rtot = 6.0f * (float)(s.S.chainD1 / n);        // Rouse: p_dir = D1 tau / h^2 / n
        if (rtot > 1.0f) rtot = 1.0f;
        if (s.rng.uf() >= rtot) continue;
        int d = (int)(6.0f * s.rng.uf()); if (d > 5) d = 5;
        dest.resize(n);
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            const int dv = s.snb(c.vox[i], FX[d], FY[d], FZ[d]);
            if (dv < 0 || (s.l1[dv] && (int)s.l1[dv] != cid)) ok = false;
            else dest[i] = dv;
        }
        if (!ok) continue;
        if (chainFrictionThin(s, c.vox.data(), dest.data(), n)) continue;
        if (beadBodyBlocksMove(s, cid, dest.data(), n)) continue;
        const double dU = sumVex(s, cid, c.species, c.vox.data(), dest.data(), n);
        if (!metropolis(s, dU, s.rng.uf())) continue;
        relocate(s, (int)k, dest);
        c.com[0] += FX[d]; c.com[1] += FY[d]; c.com[2] += FZ[d];
    }
}

// ---- interior moves -------------------------------------------------------------------------------
struct Chain { Sim& s; int k, cid, m, species; std::vector<int>& vox; std::vector<int> dir; float kappa; };

static int endBond(Chain& C, bool posTip, float d2, float ua) {
    Sim& s = C.s; const int m = C.m;
    const int moveIdx = posTip ? m - 1 : 0, pivotIdx = posTip ? m - 2 : 1;
    const int oldDir = posTip ? C.dir[m - 2] : C.dir[0];
    int nd = (int)(26.0f * d2); if (nd > 25) nd = 25;
    if (nd == oldDir) return 0;
    int dx, dy, dz; l26Dir(nd, dx, dy, dz);
    const int w = posTip ? s.snb(C.vox[pivotIdx], dx, dy, dz) : s.snb(C.vox[pivotIdx], -dx, -dy, -dz);
    if (w < 0 || (s.l1[w] && w != C.vox[moveIdx])) return 0;
    if (chainFrictionThin(s, &C.vox[moveIdx], &w, 1)) return 0;
    if (subunitHitsBody(s, C.cid, w)) return 0;
    float dE = 0.0f;
    if (m >= 3) dE = posTip ? jointBend(C.dir[m - 3], nd, C.kappa) - jointBend(C.dir[m - 3], oldDir, C.kappa)
                            : jointBend(nd, C.dir[1], C.kappa) - jointBend(oldDir, C.dir[1], C.kappa);
    dE += bondStretch(s.S, nd) - bondStretch(s.S, oldDir);
    const double dU = dE + vexInsertDelta(s, C.cid, C.species, C.vox[moveIdx], w);
    if (!metropolis(s, dU, ua)) return 0;
    const uint8_t rf = s.l3[C.vox[moveIdx]];
    clearVoxel(s, C.vox[moveIdx]);
    s.l1[w] = (uint32_t)C.cid; s.l3[w] = rf;
    if (posTip) { s.l2[w] = DIR_NONE; s.l2[C.vox[pivotIdx]] = (uint8_t)nd; C.dir[m - 2] = nd; }
    else { s.l2[w] = (uint8_t)nd; C.dir[0] = nd; }
    C.vox[moveIdx] = w;
    return 1;
}

static inline bool isEdgeVec(int x, int y, int z) {
    return ((x != 0) + (y != 0) + (z != 0)) == 2 && x >= -1 && x <= 1 && y >= -1 && y <= 1 && z >= -1 && z <= 1;
}
static inline int l26Index(int x, int y, int z) { const int r = (x + 1) + 3 * (y + 1) + 9 * (z + 1); return r < 13 ? r : r - 1; }

// structFullKinkPick: every edge vector e from the unit's - anchor whose complement to the + anchor is
// an edge vector too, less the current position; the flip is the member e = b (mask bit 2), the others
// are the sideways destinations (mask bit 8). Uniform over the admitted members.
static int fullKinkPick(int aDir, int bDir, int mask, float u, int& eOut, int& fOut) {
    if (aDir == bDir) return 0;
    int ax, ay, az, bx, by, bz; l26Dir(aDir, ax, ay, az); l26Dir(bDir, bx, by, bz);
    const int sx = ax + bx, sy = ay + by, sz = az + bz;
    int cand[4], candF[4], n = 0;
    for (int d = 0; d < 26 && n < 4; ++d) {
        int ex, ey, ez; l26Dir(d, ex, ey, ez);
        if (!isEdgeVec(ex, ey, ez) || d == aDir) continue;
        const int fx = sx - ex, fy = sy - ey, fz = sz - ez;
        if (!isEdgeVec(fx, fy, fz)) continue;
        const bool flip = (d == bDir);
        if (flip ? !(mask & 2) : !(mask & 8)) continue;
        cand[n] = d; candF[n] = l26Index(fx, fy, fz); ++n;
    }
    if (!n) return 0;
    int k = (int)((float)n * u); if (k >= n) k = n - 1;
    eOut = cand[k]; fOut = candF[k];
    return n;
}

static int kink(Chain& C, int i, int mask, float d2, float ua) {
    Sim& s = C.s; const int m = C.m;
    const int aDir = C.dir[i - 1], bDir = C.dir[i];
    int eDir, fDir;
    if (!fullKinkPick(aDir, bDir, mask, d2, eDir, fDir)) return 0;
    int ex, ey, ez; l26Dir(eDir, ex, ey, ez);
    const int w = s.snb(C.vox[i - 1], ex, ey, ez);
    if (w < 0 || s.l1[w]) return 0;
    if (chainFrictionThin(s, &C.vox[i], &w, 1)) return 0;
    if (subunitHitsBody(s, C.cid, w)) return 0;
    float dE = 0.0f;
    if (i >= 2) dE += jointBend(C.dir[i - 2], eDir, C.kappa) - jointBend(C.dir[i - 2], aDir, C.kappa);
    if (i <= m - 3) dE += jointBend(fDir, C.dir[i + 1], C.kappa) - jointBend(bDir, C.dir[i + 1], C.kappa);
    if (eDir != bDir) dE += jointBend(eDir, fDir, C.kappa) - jointBend(aDir, bDir, C.kappa);   // sideways: the joint turns
    const double dU = dE + vexInsertDelta(s, C.cid, C.species, C.vox[i], w);
    if (!metropolis(s, dU, ua)) return 0;
    const uint8_t rf = s.l3[C.vox[i]];
    clearVoxel(s, C.vox[i]);
    s.l1[w] = (uint32_t)C.cid; s.l3[w] = rf;
    s.l2[C.vox[i - 1]] = (uint8_t)eDir; s.l2[w] = (uint8_t)fDir;
    C.dir[i - 1] = eDir; C.dir[i] = fDir;
    C.vox[i] = w;
    return 1;
}

// struct_edge_translation_lin_dev.cuh: a segment of S bonds (3..6) between anchors vox[p] and vox[q]
// has its S-1 interior units displaced by one of the 12 edge vectors t, admissible when both anchor
// bonds stay edge bonds; the internal bonds are unchanged, so only the four joints at and next to the
// anchors change their bend.
static int edgeTranslation(Chain& C, float up, float usp, float ut, float ua) {
    Sim& s = C.s; const int m = C.m;
    const int SMAX = 6;                                   // STRUCT_CRANK_SMAX
    int S = 3 + (int)((float)(SMAX - 2) * usp); if (S > SMAX) S = SMAX;
    if (S > m - 1) return 0;
    const int pmax = m - 1 - S;
    int p = (int)((float)(pmax + 1) * up); if (p > pmax) p = pmax;
    const int q = p + S, nMoved = S - 1;
    int tk = (int)(12.0f * ut); if (tk > 11) tk = 11;
    int tDir = -1;
    for (int d = 0, n = 0; d < 26; ++d) { int x, y, z; l26Dir(d, x, y, z); if (isEdgeVec(x, y, z) && n++ == tk) { tDir = d; break; } }
    int tx, ty, tz, ax, ay, az, zx, zy, zz;
    l26Dir(tDir, tx, ty, tz); l26Dir(C.dir[p], ax, ay, az); l26Dir(C.dir[q - 1], zx, zy, zz);
    if (!isEdgeVec(ax + tx, ay + ty, az + tz) || !isEdgeVec(zx - tx, zy - ty, zz - tz)) return 0;
    const int newP = l26Index(ax + tx, ay + ty, az + tz), newQ = l26Index(zx - tx, zy - ty, zz - tz);
    int oldv[SMAX], neww[SMAX];
    for (int k = 1; k < S; ++k) {
        oldv[k - 1] = C.vox[p + k];
        neww[k - 1] = s.snb(C.vox[p + k], tx, ty, tz);
        if (neww[k - 1] < 0) return 0;
    }
    for (int k = 0; k < nMoved; ++k) {
        if (!s.l1[neww[k]]) continue;
        bool vacated = false;
        for (int j = 0; j < nMoved; ++j) if (oldv[j] == neww[k]) { vacated = true; break; }
        if (!vacated) return 0;
    }
    if (chainFrictionThin(s, oldv, neww, nMoved)) return 0;
    if (beadBodyBlocksMove(s, C.cid, neww, nMoved)) return 0;
    const int inP = (p >= 1) ? C.dir[p - 1] : -1, outQ = (q <= m - 2) ? C.dir[q] : -1;
    float dE = 0.0f;
    if (inP >= 0) dE += jointBend(inP, newP, C.kappa) - jointBend(inP, C.dir[p], C.kappa);
    dE += jointBend(newP, C.dir[p + 1], C.kappa) - jointBend(C.dir[p], C.dir[p + 1], C.kappa);
    dE += jointBend(C.dir[q - 2], newQ, C.kappa) - jointBend(C.dir[q - 2], C.dir[q - 1], C.kappa);
    if (outQ >= 0) dE += jointBend(newQ, outQ, C.kappa) - jointBend(C.dir[q - 1], outQ, C.kappa);
    const double dU = dE + sumVex(s, C.cid, C.species, oldv, neww, nMoved);
    if (!metropolis(s, dU, ua)) return 0;
    uint8_t rfm[SMAX];
    for (int k = 0; k < nMoved; ++k) rfm[k] = s.l3[oldv[k]];
    for (int k = 0; k < nMoved; ++k) {
        bool isDest = false;
        for (int j = 0; j < nMoved; ++j) if (neww[j] == oldv[k]) { isDest = true; break; }
        if (!isDest) clearVoxel(s, oldv[k]);
    }
    for (int k = 1; k < S; ++k) {
        s.l1[neww[k - 1]] = (uint32_t)C.cid; s.l3[neww[k - 1]] = rfm[k - 1];
        s.l2[neww[k - 1]] = (uint8_t)(k == S - 1 ? newQ : C.dir[p + k]);
        C.vox[p + k] = neww[k - 1];
    }
    s.l2[C.vox[p]] = (uint8_t)newP;
    C.dir[p] = newP; C.dir[q - 1] = newQ;
    return 1;
}

// The device's structInteriorMaskC. The SI set is 0x1B: 1 end-bond rotation, 2 flip, 8 sideways,
// 16 edge translation (bit 4, the crankshaft, is not in the set and is not ported). With the edge
// translation armed each attempt first draws which of the two families it belongs to; the other
// family then splits in thirds as the device's does, and the third that belonged to the crankshaft
// is an attempt that does nothing, exactly as on the device. LMCPU_INTERIOR_MASK overrides 0x1B (a
// diagnostic, never for timing).
static int interiorMask() {
    static int m = -1;
    if (m < 0) { const char* e = getenv("LMCPU_INTERIOR_MASK"); m = e ? (int)strtol(e, nullptr, 0) : 0x1B; }
    return m;
}

void chainInteriorAll(Sim& s) {
    const int mask = interiorMask();
    for (size_t k = 0; k < s.cx.size(); ++k) {
        Complex& c = s.cx[k];
        if (!isChain(c) || c.vox.size() < 2) continue;
        Chain C{s, (int)k, (int)k + 1, (int)c.vox.size(), c.species, c.vox, {}, (float)s.S.kappa};
        C.dir.resize(C.m);
        for (int i = 0; i < C.m; ++i) C.dir[i] = (i + 1 < C.m) ? (int)s.l2[c.vox[i]] : -1;
        for (int a = 0; a < s.S.interiorAttempts; ++a) {
            const float sel = s.rng.uf(), d1 = s.rng.uf(), d2 = s.rng.uf(), ua = s.rng.uf(), dg = s.rng.uf();
            (void)dg;                                          // the crankshaft's draw: still drawn, as on the device
            if (mask & 16) {
                const float usel = s.rng.uf();
                if (!(mask & (1 | 2 | 8)) || usel < 0.5f) {
                    const float up = s.rng.uf(), usp = s.rng.uf(), ut = s.rng.uf();
                    edgeTranslation(C, up, usp, ut, ua);
                    continue;
                }
            }
            const int cls = (sel < 1.0f / 3.0f) ? 0 : (sel < 2.0f / 3.0f) ? 1 : 2;
            if (cls == 0) { if (mask & 1) endBond(C, d1 < 0.5f, d2, ua); }
            else if (cls == 1) {
                if (!(mask & (2 | 8)) || C.m < 3) continue;
                int i = 1 + (int)((float)(C.m - 2) * d1); if (i > C.m - 2) i = C.m - 2;
                kink(C, i, mask, d2, ua);
            }
            // cls 2: the crankshaft class, not in the SI set
        }
        chainCom(s, (int)k);
    }
}
