// body.cpp -- the compact depositing body's sub-voxel step (struct_body_cell_dev.cuh
// structBodySubVoxelStep), the hard cores it is checked against (struct_body_pair_dev.cuh,
// struct_bead_body_dev.cuh) and the collisional friction shell (struct_friction_dev.cuh).
//
// The centre lives on a 1/Q sub-lattice and steps by 1/Q at Q^2 the rate, split into nrep repetitions
// so each carries a move probability of at most 1/2. A proposal leaving [0, N-1] is rejected like a
// wall; one onto an overlay voxel held by another complex is rejected; one within sigma of another
// body's centre, or within contact of a structured subunit, is rejected. The bath acts only through
// the friction factor m = 1/(1 + (F_src + F_dst)/2), which thins the attempt before its energy. The
// move energy is the smear interaction against the static baseline, identically zero for an empty
// baseline (the GPU returns before evaluating it), plus the one-body fields, zero in both scenes: so a
// surviving attempt is accepted. On the device the bodies are stepped in eight checkerboard launches
// over a randomly shifted grid; one core sweeps them in a fresh random order every step instead.
#include "sim.h"
#include "structure.h"
#include <algorithm>
#include <cmath>

// Mean over the layer just outside the smear support (max-norm distance R1) of the bath's
// friction-weighted packing sum_t c_t dxi_t[3], from the live lattice.
double frictionShellMean(const Sim& s, int v0, bool chainCoef) {
    const Scene& S = s.S;
    const int R1 = S.smearR + 1;
    int cx, cy, cz; s.coords(v0, cx, cy, cz);
    double acc = 0.0; int nIn = 0;
    for (int ax = -R1; ax <= R1; ++ax)
    for (int ay = -R1; ay <= R1; ++ay)
    for (int az = -R1; az <= R1; ++az) {
        const int m = std::max(std::abs(ax), std::max(std::abs(ay), std::abs(az)));
        if (m != R1) continue;
        const int x = cx + ax, y = cy + ay, z = cz + az;
        if (x < 0 || y < 0 || z < 0 || x >= S.NX || y >= S.NY || z >= S.NZ) continue;
        const int v = s.vid(x, y, z);
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        for (int i = 0; i < s.cnt[v]; ++i) {
            const Species& sp = S.sp[p[i]];
            acc += (double)(chainCoef ? sp.fricChain : sp.fric) * sp.dxi[3];
        }
        ++nIn;
    }
    return nIn ? acc / (double)nIn : 0.0;
}

static inline bool isBody(const Sim& s, const Complex& o) {
    return o.active && o.body && o.species == s.S.bodySpecies;
}

// One neighbourhood pass for both hard cores of a proposed body centre (nx, ny, nz).
bool bodyBlocked(const Sim& s, int cid, double nx, double ny, double nz, int centreVox) {
    const Scene& S = s.S;
    const double sp = S.smearSigmaNm / S.h, sp2 = sp * sp;
    const double sb = 0.5 * (S.smearSigmaNm + S.subunitSigmaNm) / S.h, sb2 = sb * sb;
    const int Rp = S.bodyPair ? (int)std::ceil(sp + 0.5) : 0;
    const int Rb = S.beadBody ? (int)std::ceil(sb + 0.5) : 0;
    const int R = std::max(Rp, Rb);
    if (!R) return false;
    for (int dz = -R; dz <= R; ++dz)
    for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
        const int nb = s.snb(centreVox, dx, dy, dz);
        if (nb < 0) continue;
        const uint32_t other = s.l1[nb];
        if (!other || (int)other == cid) continue;
        const Complex& o = s.cx[other - 1];
        if (!o.active) continue;
        if (isBody(s, o)) {
            if (!S.bodyPair) continue;
            const double a = o.com[0] - nx, b = o.com[1] - ny, c = o.com[2] - nz;
            if (a * a + b * b + c * c < sp2) return true;
        } else {
            if (!S.beadBody) continue;
            int vx, vy, vz; s.coords(nb, vx, vy, vz);
            const double a = nx - vx, b = ny - vy, c = nz - vz;
            if (a * a + b * b + c * c < sb2) return true;
        }
    }
    return false;
}

// The chain side of the bead-body core: would a subunit of complex cid at voxel v overlap a body?
bool subunitHitsBody(const Sim& s, int cid, int v) {
    const Scene& S = s.S;
    if (!S.beadBody) return false;
    const double sb = 0.5 * (S.smearSigmaNm + S.subunitSigmaNm) / S.h, sb2 = sb * sb;
    const int R = (int)std::ceil(sb + 0.5);
    int x, y, z; s.coords(v, x, y, z);
    for (int dz = -R; dz <= R; ++dz)
    for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx) {
        const int nb = s.snb(v, dx, dy, dz);
        if (nb < 0) continue;
        const uint32_t other = s.l1[nb];
        if (!other || (int)other == cid) continue;
        const Complex& o = s.cx[other - 1];
        if (!isBody(s, o)) continue;
        const double a = o.com[0] - x, b = o.com[1] - y, c = o.com[2] - z;
        if (a * a + b * b + c * c < sb2) return true;
    }
    return false;
}

static void bodyStep(Sim& s, int k) {
    const Scene& S = s.S;
    Complex& c = s.cx[k];
    const int cid = k + 1;
    const int Q = S.Q;
    const double delta = 1.0 / (double)Q;
    const float p_sub = (float)S.bodyPdir * (float)(Q * Q);
    int nrep = 1;
    while (6.0f * p_sub / (float)nrep > 0.5f && nrep < 64) nrep *= 2;
    const float p_rep = p_sub / (float)nrep;
    int vox0 = c.vox[0];
    double fSrc = 0.0; bool fSrcKnown = false;
    static const int SVX[6] = {1, -1, 0, 0, 0, 0}, SVY[6] = {0, 0, 1, -1, 0, 0}, SVZ[6] = {0, 0, 0, 0, 1, -1};
    for (int rep = 0; rep < nrep; ++rep) {
        if (s.rng.uf() >= 6.0f * p_rep) continue;
        int d = (int)(6.0f * s.rng.uf()); if (d > 5) d = 5;
        const double nx = c.com[0] + SVX[d] * delta, ny = c.com[1] + SVY[d] * delta, nz = c.com[2] + SVZ[d] * delta;
        if (nx < 0.0 || ny < 0.0 || nz < 0.0 || nx > S.NX - 1 || ny > S.NY - 1 || nz > S.NZ - 1) continue;
        const int rx = (int)std::floor(nx + 0.5) - (int)std::floor(c.com[0] + 0.5);
        const int ry = (int)std::floor(ny + 0.5) - (int)std::floor(c.com[1] + 0.5);
        const int rz = (int)std::floor(nz + 0.5) - (int)std::floor(c.com[2] + 0.5);
        const bool relocate = rx || ry || rz;
        int dest = vox0;
        if (relocate) {
            dest = s.snb(vox0, rx, ry, rz);
            if (dest < 0) continue;
            if (s.l1[dest] && (int)s.l1[dest] != cid) continue;
        }
        if (bodyBlocked(s, cid, nx, ny, nz, dest)) continue;
        if (S.friction) {
            if (!fSrcKnown) { fSrc = frictionShellMean(s, vox0, false); fSrcKnown = true; }
            const double fDst = relocate ? frictionShellMean(s, dest, false) : fSrc;
            const double m = 1.0 / (1.0 + 0.5 * (fSrc + fDst));
            if (m < 1.0 && s.rng.ud() >= m) continue;
            if (relocate) fSrc = fDst;
        }
        // dU = 0: empty smear baseline and zero one-body fields, so Metropolis accepts
        if (relocate) {
            clearVoxel(s, vox0);
            s.l1[dest] = (uint32_t)cid; s.l2[dest] = DIR_NONE; s.l3[dest] = (uint8_t)((ROLE_BODY << 4) | 1);
            vox0 = dest; c.vox[0] = dest;
        }
        c.com[0] = nx; c.com[1] = ny; c.com[2] = nz;
    }
}

void bodyStepAll(Sim& s) {
    static std::vector<int> order;
    order.clear();
    for (size_t k = 0; k < s.cx.size(); ++k)
        if (isBody(s, s.cx[k]) && s.cx[k].vox.size() == 1) order.push_back((int)k);
    for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[s.rng.below((uint32_t)i)]);
    for (int k : order) bodyStep(s, k);
}
