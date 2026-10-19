// cpu_drift_rdme.cpp — single-file CPU reference implementation of the driftMpdRdme update rule
// for the cross-engine benchmark (openspec change add-cross-engine-benchmark).
//
// Implements, per timestep, the same operator-split update as the GPU DriftMpdRdmeSolver for the
// reacting drift-diffusion test problem A + B <=> C in a sinusoidal one-body field:
//   X, Y, Z diffusion passes: per particle, exact-hop Poisson tau-leap on the WPE rates
//     pm = q * B(dPhi_- + lam*dF_-), pp = q * B(dPhi_+ + lam*dF_+), q = D dt / h^2,
//     B(u) = u / (e^u - 1), dF the SELF-EXCLUDED White Bear hop work, then
//     scale = (1 - e^-(pm+pp))/(pm+pp) and a cumulative split on one uniform draw
//     (byte_diffusion_drift_dev.cu, exactHopC path, tauScale = 1).
//   Reaction pass: per site, a*tau = dt*(k_f nA nB + k_off nC), fire with 1 - e^-(a tau),
//     select by propensity ratio, accept with the Froehner-Noe factor min(1, e^-dPhi_r) where
//     dPhi_r = sum_s dnu_s phi_s(v) + lam*[bFex(xi + S dxi) - bFex(xi)]
//     (byte_reaction_drift_dev.cu, precomp_reaction_kernel_fn), k_f = c_off * K / V_sub.
// The FtsZ structure machinery, charge-hole correction, smeared fields, and adaptive tau are NOT
// ported (all are off in the benchmark configuration; tau_scale == 1 was confirmed on the GPU).
// RNG is a counter-based splitmix64 hash of (step, pass, site, slot): statistically equivalent
// streams, NOT bit-identical to the GPU — validation is statistical by design.
//
// Build:  g++ -O3 -march=native -fopenmp -o cpu_drift_rdme cpu_drift_rdme.cpp
// Run:    OMP_NUM_THREADS=1 taskset -c 100 ./cpu_drift_rdme --dims 32,8,8 --n0 2048 \
//             --steps1 200 --steps2 1000 --out bench.json
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

static const int CAP = 64;              // matches MPD_LATTICE_MAX_OCCUPANCY
static const double PI = 3.14159265358979323846;

// ----------------------------------------------------------------- RNG
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline float urand(uint64_t step, uint64_t pass, uint64_t site, uint64_t slot,
                          uint64_t seed) {
    uint64_t h = splitmix64(seed ^ splitmix64(step * 0x100000001B3ULL + pass));
    h = splitmix64(h ^ splitmix64(site * 0x1000193ULL + slot));
    return (float)((h >> 11) * (1.0 / 9007199254740992.0));   // 53-bit mantissa uniform in [0,1)
}

// ----------------------------------------------------------------- physics helpers
static inline float bernoulli(float u) {
    if (u > 50.0f) u = 50.0f;
    if (u < -50.0f) u = -50.0f;
    if (std::fabs(u) < 1.0e-6f) return 1.0f - 0.5f * u + (1.0f / 12.0f) * u * u;
    return u / std::expm1(u);
}

struct Params {
    int NX = 32, NY = 8, NZ = 8;
    double h = 10.0;                 // nm
    double D = 1e5;                  // nm^2/s
    double dt = 50e-6;               // s
    double gamma[3] = {0.3, 0.3, 0.3};
    double wavelength = 80.0;        // nm
    double K = 200.0;                // nm^3
    double c_off = 5e5;              // 1/s
    double sigma[3] = {3.0, 3.0, 3.78};
    bool exclusion = true;
    uint64_t seed = 0;
};

struct Sim {
    Params P;
    int M;
    double Vsub, q, kf;
    std::vector<uint8_t> lat, nxt;   // M*CAP species bytes (0 = empty, 1..3 = A,B,C)
    std::vector<uint8_t> cnt, ncnt;  // occupancy per site
    std::vector<uint8_t> cho;        // per-slot move choice for the current pass
    std::vector<float> phi[3];       // per-species field, length M (site-node convention)
    double dxi[4][4];                // dxi[species 1..3][k]; row 0 unused
    long overflow = 0;

    // White Bear excess free energy of one voxel from weighted densities (bFex_wb port)
    double bFex(const double x[4]) const {
        double x3 = x[3];
        if (x3 <= 0.0) return 0.0;
        if (x3 > 1.0 - 1e-9) x3 = 1.0 - 1e-9;
        const double inv_pi = 0.31830988618379067154;
        const double omx = 1.0 - x3, lnomx = std::log(omx);
        return Vsub * (-(6.0 * inv_pi) * x[0] * lnomx
                       + (18.0 * inv_pi) * x[1] * x[2] / omx
                       + (6.0 * inv_pi) * (x[2] * x[2] * x[2]) / (x3 * x3)
                         * (x3 / (omx * omx) + lnomx));
    }
    void xiAt(int v, double xi[4]) const {
        xi[0] = xi[1] = xi[2] = xi[3] = 0.0;
        const uint8_t* s = &lat[(size_t)v * CAP];
        for (int i = 0; i < cnt[v]; ++i) {
            const double* d = dxi[s[i]];
            xi[0] += d[0]; xi[1] += d[1]; xi[2] += d[2]; xi[3] += d[3];
        }
    }
    // self-excluded White Bear hop work for one particle of species t, src -> dst
    double hopDelta(int t, const double xs[4], const double xd[4]) const {
        const double* d = dxi[t];
        double a[4] = {xd[0] + d[0], xd[1] + d[1], xd[2] + d[2], xd[3] + d[3]};
        double b[4] = {xs[0] - d[0], xs[1] - d[1], xs[2] - d[2], xs[3] - d[3]};
        return (bFex(a) - bFex(xd)) - (bFex(xs) - bFex(b));
    }

    void init() {
        M = P.NX * P.NY * P.NZ;
        Vsub = P.h * P.h * P.h;
        q = P.D * P.dt / (P.h * P.h);
        kf = P.c_off * P.K / Vsub;
        lat.assign((size_t)M * CAP, 0); nxt = lat;
        cnt.assign(M, 0); ncnt = cnt;
        cho.assign((size_t)M * CAP, 0);
        for (int s = 0; s < 3; ++s) {
            phi[s].resize(M);
            for (int v = 0; v < M; ++v) {
                int x = v % P.NX;
                phi[s][v] = (float)(P.gamma[s]
                            * std::sin(2.0 * PI * (double)x * P.h / P.wavelength));
            }
            for (int k = 0; k < 4; ++k)
                dxi[s + 1][k] = P.exclusion
                    ? (PI / (6.0 * Vsub)) * std::pow(P.sigma[s], (double)k) : 0.0;
        }
    }
    void place(int species, long n, uint64_t salt) {
        for (long i = 0; i < n; ++i) {
            for (int tries = 0;; ++tries) {
                int v = (int)(splitmix64(P.seed ^ salt ^ (uint64_t)(i * 1000003L + tries)) % M);
                if (cnt[v] < CAP) { lat[(size_t)v * CAP + cnt[v]++] = (uint8_t)species; break; }
            }
        }
    }

    // one diffusion pass along `axis` (0=X fastest index, 1=Y, 2=Z), GPU choice-then-gather order
    void diffusionPass(int axis, uint64_t step) {
        const int NX = P.NX, NY = P.NY, NZ = P.NZ;
        const bool excl = P.exclusion;
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < M; ++v) {
            if (!cnt[v]) continue;
            int x = v % NX, y = (v / NX) % NY, z = v / (NX * NY);
            int vm, vp;
            if (axis == 0) { vm = (x ? x - 1 : NX - 1) + NX * (y + NY * z);
                             vp = (x + 1 < NX ? x + 1 : 0) + NX * (y + NY * z); }
            else if (axis == 1) { vm = x + NX * ((y ? y - 1 : NY - 1) + NY * z);
                                  vp = x + NX * ((y + 1 < NY ? y + 1 : 0) + NY * z); }
            else { vm = x + NX * (y + NY * (z ? z - 1 : NZ - 1));
                   vp = x + NX * (y + NY * (z + 1 < NZ ? z + 1 : 0)); }
            double xs[4], xm[4], xp[4];
            if (excl) { xiAt(v, xs); xiAt(vm, xm); xiAt(vp, xp); }
            const uint8_t* s = &lat[(size_t)v * CAP];
            uint8_t* c = &cho[(size_t)v * CAP];
            for (int i = 0; i < cnt[v]; ++i) {
                const int t = s[i], sp = t - 1;
                float dum = phi[sp][vm] - phi[sp][v];
                float dup = phi[sp][vp] - phi[sp][v];
                if (excl) {
                    dum += (float)hopDelta(t, xs, xm);
                    dup += (float)hopDelta(t, xs, xp);
                }
                float pm = (float)q * bernoulli(dum);
                float pp = (float)q * bernoulli(dup);
                const float sum = pm + pp;
                if (sum > 0.0f) {
                    const float scale = (1.0f - std::exp(-sum)) / sum;
                    pm *= scale; pp *= scale;
                }
                const float r = urand(step, axis, (uint64_t)v, (uint64_t)i, P.seed);
                c[i] = (r < pm) ? 1 : (r < pm + pp) ? 2 : 0;   // minus / plus / stay
            }
        }
        // gather: destination site collects its stayers and both neighbors' movers-in
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < M; ++v) {
            int x = v % NX, y = (v / NX) % NY, z = v / (NX * NY);
            int vm, vp;   // vm = my minus-side neighbor (its "plus" movers arrive here)
            if (axis == 0) { vm = (x ? x - 1 : NX - 1) + NX * (y + NY * z);
                             vp = (x + 1 < NX ? x + 1 : 0) + NX * (y + NY * z); }
            else if (axis == 1) { vm = x + NX * ((y ? y - 1 : NY - 1) + NY * z);
                                  vp = x + NX * ((y + 1 < NY ? y + 1 : 0) + NY * z); }
            else { vm = x + NX * (y + NY * (z ? z - 1 : NZ - 1));
                   vp = x + NX * (y + NY * (z + 1 < NZ ? z + 1 : 0)); }
            uint8_t* out = &nxt[(size_t)v * CAP];
            int n = 0;
            const uint8_t* s0 = &lat[(size_t)v * CAP]; const uint8_t* c0 = &cho[(size_t)v * CAP];
            for (int i = 0; i < cnt[v]; ++i) if (c0[i] == 0) out[n++] = s0[i];
            const uint8_t* sm = &lat[(size_t)vm * CAP]; const uint8_t* cm = &cho[(size_t)vm * CAP];
            for (int i = 0; i < cnt[vm]; ++i)
                if (cm[i] == 2) { if (n < CAP) out[n++] = sm[i]; else ++overflow; }
            const uint8_t* sp_ = &lat[(size_t)vp * CAP]; const uint8_t* cp = &cho[(size_t)vp * CAP];
            for (int i = 0; i < cnt[vp]; ++i)
                if (cp[i] == 1) { if (n < CAP) out[n++] = sp_[i]; else ++overflow; }
            ncnt[v] = (uint8_t)n;
            for (int i = n; i < CAP; ++i) out[i] = 0;
        }
        lat.swap(nxt); cnt.swap(ncnt);
    }

    void reactionPass(uint64_t step) {
        const bool excl = P.exclusion;
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < M; ++v) {
            uint8_t* s = &lat[(size_t)v * CAP];
            int nA = 0, nB = 0, nC = 0;
            for (int i = 0; i < cnt[v]; ++i) {
                if (s[i] == 1) ++nA; else if (s[i] == 2) ++nB; else ++nC;
            }
            const double aF = kf * (double)nA * (double)nB * P.dt;   // fusion  a*tau
            const double aR = P.c_off * (double)nC * P.dt;           // fission a*tau
            const double at = aF + aR;
            if (at <= 0.0) continue;
            const float Pfire = (at > 2e-4) ? (float)(1.0 - std::exp(-at)) : (float)at;
            if (urand(step, 3, (uint64_t)v, 0, P.seed) >= Pfire) continue;
            const bool fusion = urand(step, 4, (uint64_t)v, 0, P.seed) * at < aF;
            // Froehner-Noe acceptance: field term + whole-voxel White Bear reaction work
            float dPhi = fusion ? (phi[2][v] - phi[0][v] - phi[1][v])
                                : (phi[0][v] + phi[1][v] - phi[2][v]);
            if (excl) {
                double xi[4]; xiAt(v, xi);
                double xn[4];
                for (int k = 0; k < 4; ++k) {
                    const double dC = dxi[3][k], dA = dxi[1][k], dB = dxi[2][k];
                    xn[k] = xi[k] + (fusion ? (dC - dA - dB) : (dA + dB - dC));
                }
                dPhi += (float)(bFex(xn) - bFex(xi));
            }
            if (dPhi > 50.0f) dPhi = 50.0f;
            if (dPhi < -50.0f) dPhi = -50.0f;
            const float pi_r = std::fmin(1.0f, std::exp(-dPhi));
            if (urand(step, 5, (uint64_t)v, 0, P.seed) > pi_r) continue;
            if (fusion) {
                if (!nA || !nB) continue;
                // one A becomes C, one B is removed, survivors packed densely (GPU order)
                uint8_t tmp[CAP]; int w = 0; bool doneA = false, doneB = false;
                for (int i = 0; i < cnt[v]; ++i) {
                    uint8_t sp = s[i];
                    if (sp == 1 && !doneA) { sp = 3; doneA = true; }
                    else if (sp == 2 && !doneB) { doneB = true; continue; }
                    tmp[w++] = sp;
                }
                std::memcpy(s, tmp, w); std::memset(s + w, 0, CAP - w);
                cnt[v] = (uint8_t)w;
            } else {
                if (!nC) continue;
                if (cnt[v] + 1 > CAP) { ++overflow; continue; }
                for (int i = 0; i < cnt[v]; ++i) if (s[i] == 3) { s[i] = 1; break; }  // C -> A
                s[cnt[v]++] = 2;                                                     // insert B
            }
        }
    }

    void step(uint64_t t) {
        diffusionPass(0, 4 * t + 0);
        diffusionPass(1, 4 * t + 1);
        diffusionPass(2, 4 * t + 2);
        reactionPass(4 * t + 3);
    }
};

int main(int argc, char** argv) {
    Params P;
    long n0 = 2048, steps1 = 200, steps2 = 1000, obsEvery = 0, obsFrom = 0;
    std::string out = "cpu_bench.json", profOut;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--dims") sscanf(next().c_str(), "%d,%d,%d", &P.NX, &P.NY, &P.NZ);
        else if (a == "--n0") n0 = atol(next().c_str());
        else if (a == "--gamma") { double g; g = atof(next().c_str());
                                   P.gamma[0] = P.gamma[1] = P.gamma[2] = g; }
        else if (a == "--wavelength") P.wavelength = atof(next().c_str());
        else if (a == "--K") P.K = atof(next().c_str());
        else if (a == "--c-off") P.c_off = atof(next().c_str());
        else if (a == "--dt") P.dt = atof(next().c_str());
        else if (a == "--seed") P.seed = (uint64_t)atol(next().c_str());
        else if (a == "--no-exclusion") P.exclusion = false;
        else if (a == "--steps1") steps1 = atol(next().c_str());
        else if (a == "--steps2") steps2 = atol(next().c_str());
        else if (a == "--observe-every") obsEvery = atol(next().c_str());
        else if (a == "--observe-from") obsFrom = atol(next().c_str());
        else if (a == "--profile-out") profOut = next();
        else if (a == "--out") out = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    Sim S; S.P = P; S.init();
    S.place(1, n0, 0xA11CEULL); S.place(2, n0, 0xB0BULL);
    int threads = 1;
    #ifdef _OPENMP
    threads = omp_get_max_threads();
    #endif

    // observation accumulators (validation mode): per-x-column A occupancy + species totals
    std::vector<double> profA(P.NX, 0.0);
    double totA = 0, totB = 0, totC = 0; long nsamp = 0;

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (long t = 0; t < steps1; ++t) S.step((uint64_t)t);
    auto t1 = clk::now();
    for (long t = steps1; t < steps1 + steps2; ++t) {
        S.step((uint64_t)t);
        if (obsEvery > 0 && t >= obsFrom && (t % obsEvery) == 0) {
            ++nsamp;
            for (int v = 0; v < S.M; ++v) {
                const uint8_t* s = &S.lat[(size_t)v * CAP];
                for (int i = 0; i < S.cnt[v]; ++i) {
                    if (s[i] == 1) { profA[v % P.NX] += 1.0; ++totA; }
                    else if (s[i] == 2) ++totB;
                    else ++totC;
                }
            }
        }
    }
    auto t2 = clk::now();
    const double w1 = std::chrono::duration<double>(t1 - t0).count();
    const double w2 = std::chrono::duration<double>(t2 - t1).count();
    const double per_step = (w2 - w1 > 0 && steps2 != steps1)
                          ? (w2 - w1) / (double)(steps2 - steps1) : w2 / (double)steps2;

    FILE* f = fopen(out.c_str(), "w");
    fprintf(f, "{\"engine\":\"cpu_drift_rdme\",\"dims\":[%d,%d,%d],\"M\":%d,\"n0\":%ld,"
               "\"threads\":%d,\"exclusion\":%d,\"dt_s\":%g,\"steps1\":%ld,\"steps2\":%ld,"
               "\"wall1_s\":%.6f,\"wall2_s\":%.6f,\"per_step_s\":%.6g,\"sim_per_wall\":%.6g,"
               "\"overflow\":%ld,\"seed\":%llu}\n",
            P.NX, P.NY, P.NZ, S.M, n0, threads, (int)P.exclusion, P.dt, steps1, steps2,
            w1, w2, per_step, P.dt / per_step, S.overflow, (unsigned long long)P.seed);
    fclose(f);
    if (!profOut.empty() && nsamp > 0) {
        FILE* g = fopen(profOut.c_str(), "w");
        const double colSites = (double)(P.NY * P.NZ) * nsamp;
        fprintf(g, "# x_site  mean_nA_per_site   (K_eff = NC*V/(NA*NB), V = %g nm^3)\n",
                S.M * S.Vsub);
        for (int x = 0; x < P.NX; ++x) fprintf(g, "%d %.8f\n", x, profA[x] / colSites);
        fprintf(g, "# totals per sample: A %.4f B %.4f C %.4f  K_eff %.6f\n",
                totA / nsamp, totB / nsamp, totC / nsamp,
                (totC / nsamp) * (S.M * S.Vsub) / ((totA / nsamp) * (totB / nsamp)));
        fclose(g);
    }
    printf("BENCH cpu dims=(%d,%d,%d) M=%d n0=%ld threads=%d excl=%d per_step=%.4gs "
           "sim/wall=%.4g overflow=%ld\n", P.NX, P.NY, P.NZ, S.M, n0, threads,
           (int)P.exclusion, per_step, P.dt / per_step, S.overflow);
    return 0;
}
