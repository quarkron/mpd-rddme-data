// smear.cpp -- the depositing bodies' weighted-density field (byte_structure_drift_dev.cu
// structSmearDeposit, mpd_structure_smear_refresh_kernel / _incremental_kernel; StructureSmear.cu).
//
// A body's stencil sits at its centre itself, quantised to 1/1024 voxel and split over the up to
// eight corners of its cell with cloud-in-cell weights; each corner's stencil is clipped at the faces
// of the (non-periodic) structure lattice. The full re-stamp resets the field to the static baseline
// and deposits every body; the incremental update moves only the bodies whose quantised centre
// changed, and is itself re-stamped in full every smearIncremental steps, as the host does.
#include "sim.h"
#include <cmath>
#include <cstring>

struct Centre { int i0[3]; double w1[3]; };

static inline void centreOf(const int q[3], Centre& c) {
    const int Q = SMEAR_COM_Q;
    for (int a = 0; a < 3; ++a) {
        const int i0 = (q[a] >= 0) ? q[a] / Q : -((-q[a] + Q - 1) / Q);
        c.i0[a] = i0;
        c.w1[a] = (double)(q[a] - i0 * Q) / (double)Q;
    }
}

static void deposit(Sim& s, const Centre& c, double sign) {
    const Scene& S = s.S;
    const int R = S.smearR, Wd = 2 * R + 1;
    const size_t M = (size_t)s.M;
    for (int kz = 0; kz < 2; ++kz)
    for (int ky = 0; ky < 2; ++ky)
    for (int kx = 0; kx < 2; ++kx) {
        const double W = sign * (kx ? c.w1[0] : 1.0 - c.w1[0]) * (ky ? c.w1[1] : 1.0 - c.w1[1])
                              * (kz ? c.w1[2] : 1.0 - c.w1[2]);
        if (W == 0.0) continue;
        const int cx = c.i0[0] + kx, cy = c.i0[1] + ky, cz = c.i0[2] + kz;
        for (int dz = -R; dz <= R; ++dz)
        for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            const int xt = cx + dx, yt = cy + dy, zt = cz + dz;
            if (xt < 0 || xt >= S.NX || yt < 0 || yt >= S.NY || zt < 0 || zt >= S.NZ) continue;
            const float* st = &S.stencil[(size_t)((((dx + R) * Wd + (dy + R)) * Wd + (dz + R)) * 4)];
            const size_t v = (size_t)s.vid(xt, yt, zt);
            s.smear[v] += W * (double)st[0];
            s.smear[M + v] += W * (double)st[1];
            s.smear[2 * M + v] += W * (double)st[2];
            s.smear[3 * M + v] += W * (double)st[3];
        }
    }
}

static inline bool depositing(const Sim& s, const Complex& c) {
    return c.active && c.body && c.species == s.S.bodySpecies;
}

void smearFullRestamp(Sim& s) {
    std::memcpy(s.smear.data(), s.smearBase.data(), s.smear.size() * sizeof(double));
    for (size_t k = 0; k < s.cx.size(); ++k) {
        const Complex& c = s.cx[k];
        if (!depositing(s, c)) continue;
        const int q[3] = {structQuant(c.com[0]), structQuant(c.com[1]), structQuant(c.com[2])};
        s.frozen[3 * k] = q[0]; s.frozen[3 * k + 1] = q[1]; s.frozen[3 * k + 2] = q[2];
        Centre cc; centreOf(q, cc);
        deposit(s, cc, +1.0);
    }
}

static void smearIncremental(Sim& s) {
    for (size_t k = 0; k < s.cx.size(); ++k) {
        const Complex& c = s.cx[k];
        if (!depositing(s, c)) continue;
        const int q[3] = {structQuant(c.com[0]), structQuant(c.com[1]), structQuant(c.com[2])};
        int* o = &s.frozen[3 * k];
        if (q[0] == o[0] && q[1] == o[1] && q[2] == o[2]) continue;
        if (o[0] >= 0 && o[1] >= 0 && o[2] >= 0) { Centre co; centreOf(o, co); deposit(s, co, -1.0); }
        Centre cn; centreOf(q, cn);
        deposit(s, cn, +1.0);
        o[0] = q[0]; o[1] = q[1]; o[2] = q[2];
    }
}

// Before the structure moves, reading the centres the previous step left.
void smearStep(Sim& s) {
    const int k = s.S.smearIncremental;
    if (k <= 0 || s.stepIndex % k == 0) smearFullRestamp(s);
    else smearIncremental(s);
}
