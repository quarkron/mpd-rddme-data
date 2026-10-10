// bath_hop.cpp -- one axis pass of the bath drift-diffusion (byte_diffusion_drift_dev.cu,
// makeDiffusionChoicesDrift and the propagation that follows it).
//
// Per voxel the weighted densities xi are the bath particles' increments, plus the structure fill
// counted as fillSpecies (reciprocal VEX), plus the bodies' smear unless the free-volume functional is
// armed. The hop energy of a type-t particle from s to d is the telescoped self-excluded White Bear
// work lambda [(F(xi_d + dxi_t) - F_d) - (F_s - F(xi_s - dxi_t))], with F the free-volume form and the
// confinement term -ln(1-w_d) + ln(1-w_s) added unscaled when that functional is armed. The one-body
// field is zero in both scenes, which the GPU also short-circuits (all gamma = 0).
//
// Choices are made for every particle from the lattice as it stood at the start of the pass, then all
// particles move at once (a Jacobi update), as on the device. The GPU evaluates the three voxels'
// densities per source cell out of a shared window; here they are computed once per voxel per pass,
// the single-core form of the same hoist (ERA 034's publish phase).
#include "sim.h"
#include "wb.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline void periodicNb(const Sim& s, int v, int axis, int& vm, int& vp) {
    const int NX = s.S.NX, NY = s.S.NY, NZ = s.S.NZ;
    const int x = v % NX, y = (v / NX) % NY, z = v / (NX * NY);
    if (axis == 0) { vm = (x ? x - 1 : NX - 1) + NX * (y + NY * z); vp = (x + 1 < NX ? x + 1 : 0) + NX * (y + NY * z); }
    else if (axis == 1) { vm = x + NX * ((y ? y - 1 : NY - 1) + NY * z); vp = x + NX * ((y + 1 < NY ? y + 1 : 0) + NY * z); }
    else { vm = x + NX * (y + NY * (z ? z - 1 : NZ - 1)); vp = x + NX * (y + NY * (z + 1 < NZ ? z + 1 : 0)); }
}

// xi, w and F of every voxel, from the lattice as it stands.
static void publishDensities(Sim& s) {
    const Scene& S = s.S;
    const float V = (float)S.V;
    const Species& fillSp = S.sp[S.fillSpecies];
    const bool foldSmear = !S.freeVol;
    for (int v = 0; v < s.M; ++v) {
        float x0 = 0, x1 = 0, x2 = 0, x3 = 0;
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        for (int i = 0; i < s.cnt[v]; ++i) {
            const float* d = S.sp[p[i]].dxif;
            x0 += d[0]; x1 += d[1]; x2 += d[2]; x3 += d[3];
        }
        if (S.reciprocal) {
            const int fill = s.fillAt(v);
            if (fill > 0) {
                const float f = (float)fill;
                x0 += f * fillSp.dxif[0]; x1 += f * fillSp.dxif[1]; x2 += f * fillSp.dxif[2]; x3 += f * fillSp.dxif[3];
            }
        }
        if (foldSmear) {
            x0 += (float)s.smear[v]; x1 += (float)s.smear[(size_t)s.M + v];
            x2 += (float)s.smear[(size_t)2 * s.M + v]; x3 += (float)s.smear[(size_t)3 * s.M + v];
        }
        float* o = &s.xi[(size_t)4 * v];
        o[0] = x0; o[1] = x1; o[2] = x2; o[3] = x3;
        if (S.freeVol) {
            const float w = fvClamp((float)s.smear[(size_t)3 * s.M + v]);
            s.wv[v] = w;
            s.Fv[v] = bFexFV(V, x0, x1, x2, x3, w);
        } else {
            s.Fv[v] = bFexWB(V, x0, x1, x2, x3);
        }
    }
}

// Hop energy of one type-t particle from src (whose removal work is rem) into dst.
static inline float hopDU(const Sim& s, const float* d, int src, int dst, float rem) {
    const float V = (float)s.S.V;
    const float* xd = &s.xi[(size_t)4 * dst];
    if (s.S.freeVol) {
        const float wd = s.wv[dst], ws = s.wv[src];
        const float ins = bFexFV(V, xd[0] + d[0], xd[1] + d[1], xd[2] + d[2], xd[3] + d[3], wd) - s.Fv[dst];
        return s.S.lambda * (ins - rem) + (-logf(1.0f - wd) + logf(1.0f - ws));
    }
    const float ins = bFexWB(V, xd[0] + d[0], xd[1] + d[1], xd[2] + d[2], xd[3] + d[3]) - s.Fv[dst];
    return s.S.lambda * (ins - rem);
}

void bathPass(Sim& s, int axis) {
    const Scene& S = s.S;
    const bool hc = S.lambda != 0.0f;
    const float V = (float)S.V;
    if (hc) publishDensities(s);
    // choices
    for (int v = 0; v < s.M; ++v) {
        const int n = s.cnt[v];
        if (!n) continue;
        int vm, vp; periodicNb(s, v, axis, vm, vp);
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        uint8_t* c = &s.choice[(size_t)v * CAP];
        const float* xs = &s.xi[(size_t)4 * v];
        for (int i = 0; i < n; ++i) {
            const Species& sp = S.sp[p[i]];
            const float q = sp.q;
            float pm = 0.0f, pp = 0.0f;
            if (q > 0.0f) {
                float dum = 0.0f, dup = 0.0f;
                if (hc) {
                    const float* d = sp.dxif;
                    const float rem = S.freeVol
                        ? s.Fv[v] - bFexFV(V, xs[0] - d[0], xs[1] - d[1], xs[2] - d[2], xs[3] - d[3], s.wv[v])
                        : s.Fv[v] - bFexWB(V, xs[0] - d[0], xs[1] - d[1], xs[2] - d[2], xs[3] - d[3]);
                    dum = hopDU(s, d, v, vm, rem);
                    dup = hopDU(s, d, v, vp, rem);
                }
                pm = q * bernoulliF(dum);
                pp = q * bernoulliF(dup);
            }
            const float r = s.rng.uf();
            if (S.exactHop) {
                const float sum = pm + pp;
                if (sum > 0.0f) { const float sc = (-expm1f(-sum)) / sum; pm *= sc; pp *= sc; }
                c[i] = (r < pm) ? 1 : (r < pm + pp) ? 2 : 0;
            } else {
                uint8_t ch = (r < pm) ? 1 : 0;
                if (r >= 0.5f && r < pp + 0.5f) ch = 2;
                c[i] = ch;
            }
        }
    }
    // propagation: every staying particle first, then the movers; a mover that finds its destination
    // full goes back to its source (counted); a source filled by arrivals as well stops the run loudly
    std::memset(s.ncnt.data(), 0, s.M);
    for (int v = 0; v < s.M; ++v) {
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        const uint8_t* c = &s.choice[(size_t)v * CAP];
        uint8_t* o = &s.nlat[(size_t)v * CAP];
        int w = 0;
        for (int i = 0; i < s.cnt[v]; ++i) if (!c[i]) o[w++] = p[i];
        s.ncnt[v] = (uint8_t)w;
    }
    for (int v = 0; v < s.M; ++v) {
        const int n = s.cnt[v];
        if (!n) continue;
        int vm, vp; periodicNb(s, v, axis, vm, vp);
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        const uint8_t* c = &s.choice[(size_t)v * CAP];
        for (int i = 0; i < n; ++i) {
            if (!c[i]) continue;
            int d = (c[i] == 1) ? vm : vp;
            if (s.ncnt[d] >= CAP) { d = v; ++s.overflow; }
            if (s.ncnt[d] >= CAP) { fprintf(stderr, "[bath] site %d and its neighbour both full: particle would be lost\n", v); exit(4); }
            s.nlat[(size_t)d * CAP + s.ncnt[d]++] = p[i];
        }
    }
    s.lat.swap(s.nlat); s.cnt.swap(s.ncnt);
}
