// wb.h -- the White Bear excess free energy of one voxel, its free-volume form, and the Bernoulli
// factor, in single precision as the GPU hop path evaluates them (byte_diffusion_drift_dev.cu
// bFex_wb_f, bernoulliFast; struct_freevol_dev.cuh bFex_fv_f).
#ifndef LMCPU_WB_H_
#define LMCPU_WB_H_
#include <cmath>

// beta F_ex of a voxel of volume V holding the scalar weighted densities x0..x3 (White Bear / BMCSL).
static inline float bFexWB(float V, float x0, float x1, float x2, float x3) {
    if (x3 <= 0.0f) return 0.0f;
    if (x3 > 1.0f - 1.0e-6f) x3 = 1.0f - 1.0e-6f;
    const float inv_pi = 0.31830988618379067154f;
    const float omx = 1.0f - x3, lnomx = logf(omx);
    const float t1 = -(6.0f * inv_pi) * x0 * lnomx;
    const float t2 = (18.0f * inv_pi) * (x1 * x2) / omx;
    const float t3 = (6.0f * inv_pi) * (x2 * x2 * x2) / (x3 * x3) * (x3 / (omx * omx) + lnomx);
    return V * (t1 + t2 + t3);
}

// The same functional in double precision, as the structure kernels evaluate it (bFex_wb).
static inline double bFexWBd(double V, double x0, double x1, double x2, double x3) {
    if (x3 <= 0.0) return 0.0;
    if (x3 > 1.0 - 1e-9) x3 = 1.0 - 1e-9;
    const double inv_pi = 0.31830988618379067154;
    const double omx = 1.0 - x3, lnomx = std::log(omx);
    return V * (-(6.0 * inv_pi) * x0 * lnomx + (18.0 * inv_pi) * x1 * x2 / omx
                + (6.0 * inv_pi) * (x2 * x2 * x2) / (x3 * x3) * (x3 / (omx * omx) + lnomx));
}

// Fraction of the voxel a body excludes from the bath, clamped as fvWeightAt does.
static inline float fvClamp(float w) {
    const float WMAX = 1.0f - 1.0f / 1024.0f;
    if (w <= 0.0f) return 0.0f;
    return w >= WMAX ? WMAX : w;
}

// The bath at its own packing in the voxel's free volume: V (1-w) f_WB(xi / (1-w)).
static inline float bFexFV(float V, float x0, float x1, float x2, float x3, float w) {
    const float f = 1.0f - w, inv = 1.0f / f;
    return f * bFexWB(V, x0 * inv, x1 * inv, x2 * inv, x3 * inv);
}

// B(u) = u / (e^u - 1), clamped at |u| = 50 and Taylor-expanded near 0 (bernoulliFast).
static inline float bernoulliF(float u) {
    if (u > 50.0f) u = 50.0f;
    if (u < -50.0f) u = -50.0f;
    if (fabsf(u) < 0.0625f) {
        const float u2 = u * u;
        return 1.0f - 0.5f * u + u2 * ((1.0f / 12.0f) - u2 * (1.0f / 720.0f));
    }
    return u / (expf(u) - 1.0f);
}
#endif
