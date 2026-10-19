// adv_cpu.cpp — general-scheme CPU reference implementation of the driftMpdRdme update rule,
// for the advantage-metric campaign.
//
// Derived from benchmark/cpu_drift_rdme.cpp, which is validated head to head against the GPU
// solver (equilibrium field response to 0.4%, equilibrium constant within two standard errors).
// Every physics function here — bernoulli, bFex, xiAt, hopDelta, the diffusion pass, the
// Froehner-Noe acceptance — is carried over unchanged. The ONLY generalization is that the
// species set and the reaction set are read from the command line instead of being hardcoded to
// A + B <=> C, so that one binary serves the purely diffusive, reversible bimolecular and
// mass-action-network schemes of the campaign. The original file is left untouched so the
// published figure keeps its provenance.
//
// Reaction channels, given as --rxn <kind>:<reactants>/<products>@<rate>, species 0-based.
// The arrow is '/' and not '>' so the argument needs no shell quoting:
//   fusion:0,1/2@1e2      A + B -> C     propensity  rate * n0 * n1
//   fission:2/0,1@2e2     C -> A + B     propensity  rate * n2
//   conversion:2/3@2e2    C -> D         propensity  rate * n2
// A fusion rate is per SUBVOLUME volume here, i.e. k_f = c_off*K/V_sub in the original; the
// driver converts from the per-system-volume convention before it builds the argument.
//
// Build:  g++ -O3 -march=native -fopenmp -o adv_cpu adv_cpu.cpp
// Run:    OMP_NUM_THREADS=1 taskset -c 40 ./adv_cpu --dims 32,8,8 --nspec 3 \
//             --sigma 3,3,3.7798 --seed-counts 0:1449,1:1449 \
//             --rxn fusion:0,1/2@0.0195 --rxn fission:2/0,1@200 \
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
static const int MAXS = 24;             // species cap of this reference (8 until 2026-09-30: the pair sweep needs 18)
static const int MAXR = 16;             // reaction-channel cap (12 until 2026-09-30)
static const double PI = 3.14159265358979323846;

// ----------------------------------------------------------------- RNG (unchanged)
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
    return (float)((h >> 11) * (1.0 / 9007199254740992.0));
}

// ----------------------------------------------------------------- physics (unchanged)
static inline float bernoulli(float u) {
    if (u > 50.0f) u = 50.0f;
    if (u < -50.0f) u = -50.0f;
    if (std::fabs(u) < 1.0e-6f) return 1.0f - 0.5f * u + (1.0f / 12.0f) * u * u;
    return u / std::expm1(u);
}

enum RKind { FUSION = 0, FISSION = 1, CONVERSION = 2 };

struct Rxn {
    RKind kind;
    int r0, r1;        // reactants, 0-based; r1 = -1 unless FUSION
    int p0, p1;        // products,  0-based; p1 = -1 unless FISSION
    double rate;       // per-subvolume rate constant
};

struct Params {
    int NX = 32, NY = 8, NZ = 8;
    int nspec = 3;
    double h = 10.0;
    double D = 1e5;
    double dt = 50e-6;
    double gamma[MAXS];
    double sigma[MAXS];
    double wavelength = 80.0;
    bool exclusion = true;
    bool stokes = true;       // scale D by smin/sigma_s; --no-stokes reproduces cpu_drift_rdme
    uint64_t seed = 0;
    int nrxn = 0;
    Rxn rxn[MAXR];
    long seedCount[MAXS];
    Params() {
        for (int i = 0; i < MAXS; ++i) { gamma[i] = 0.3; sigma[i] = 3.0; seedCount[i] = 0; }
    }
};

struct Sim {
    Params P;
    int M;
    double Vsub, q;
    std::vector<uint8_t> lat, nxt;   // M*CAP species bytes (0 = empty, 1..nspec)
    std::vector<uint8_t> cnt, ncnt;
    std::vector<uint8_t> cho;
    std::vector<float> phi[MAXS];
    double dxi[MAXS + 1][4];         // dxi[species 1..nspec][k]; row 0 unused
    long overflow = 0;
    // D scaled per species by Stokes on the diameter, as every other arm of the campaign does.
    double qs[MAXS];

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
        double smin = P.sigma[0];
        for (int s = 1; s < P.nspec; ++s) if (P.sigma[s] < smin) smin = P.sigma[s];
        lat.assign((size_t)M * CAP, 0); nxt = lat;
        cnt.assign(M, 0); ncnt = cnt;
        cho.assign((size_t)M * CAP, 0);
        for (int s = 0; s < P.nspec; ++s) {
            qs[s] = P.stokes ? q * (smin / P.sigma[s]) : q;   // --no-stokes: one D for all species
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

    void neighbors(int v, int axis, int& vm, int& vp) const {
        const int NX = P.NX, NY = P.NY, NZ = P.NZ;
        int x = v % NX, y = (v / NX) % NY, z = v / (NX * NY);
        if (axis == 0) { vm = (x ? x - 1 : NX - 1) + NX * (y + NY * z);
                         vp = (x + 1 < NX ? x + 1 : 0) + NX * (y + NY * z); }
        else if (axis == 1) { vm = x + NX * ((y ? y - 1 : NY - 1) + NY * z);
                              vp = x + NX * ((y + 1 < NY ? y + 1 : 0) + NY * z); }
        else { vm = x + NX * (y + NY * (z ? z - 1 : NZ - 1));
               vp = x + NX * (y + NY * (z + 1 < NZ ? z + 1 : 0)); }
    }

    void diffusionPass(int axis, uint64_t step) {
        const bool excl = P.exclusion;
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < M; ++v) {
            if (!cnt[v]) continue;
            int vm, vp; neighbors(v, axis, vm, vp);
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
                float pm = (float)qs[sp] * bernoulli(dum);
                float pp = (float)qs[sp] * bernoulli(dup);
                const float sum = pm + pp;
                if (sum > 0.0f) {
                    const float scale = (1.0f - std::exp(-sum)) / sum;
                    pm *= scale; pp *= scale;
                }
                const float r = urand(step, axis, (uint64_t)v, (uint64_t)i, P.seed);
                c[i] = (r < pm) ? 1 : (r < pm + pp) ? 2 : 0;
            }
        }
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < M; ++v) {
            int vm, vp; neighbors(v, axis, vm, vp);
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

    // propensity of channel r at the voxel counts n[] (0-based species index)
    static inline double propensity(const Rxn& R, const int* n) {
        switch (R.kind) {
            case FUSION:     return R.rate * (double)n[R.r0] * (double)n[R.r1];
            case FISSION:
            case CONVERSION: return R.rate * (double)n[R.r0];
        }
        return 0.0;
    }

    void reactionPass(uint64_t step) {
        if (!P.nrxn) return;
        const bool excl = P.exclusion;
        const int NS = P.nspec;
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < M; ++v) {
            uint8_t* s = &lat[(size_t)v * CAP];
            int n[MAXS]; for (int i = 0; i < NS; ++i) n[i] = 0;
            for (int i = 0; i < cnt[v]; ++i) ++n[s[i] - 1];
            double a[MAXR], at = 0.0;
            for (int r = 0; r < P.nrxn; ++r) { a[r] = propensity(P.rxn[r], n) * P.dt; at += a[r]; }
            if (at <= 0.0) continue;
            const float Pfire = (at > 2e-4) ? (float)(1.0 - std::exp(-at)) : (float)at;
            if (urand(step, 3, (uint64_t)v, 0, P.seed) >= Pfire) continue;
            // select the channel in proportion to its propensity
            double u = (double)urand(step, 4, (uint64_t)v, 0, P.seed) * at;
            int rc = P.nrxn - 1;
            for (int r = 0; r < P.nrxn; ++r) { u -= a[r]; if (u < 0.0) { rc = r; break; } }
            const Rxn& R = P.rxn[rc];
            // stoichiometry of the chosen channel
            int dn[MAXS]; for (int i = 0; i < NS; ++i) dn[i] = 0;
            --dn[R.r0]; if (R.r1 >= 0) --dn[R.r1];
            ++dn[R.p0]; if (R.p1 >= 0) ++dn[R.p1];
            // Froehner-Noe acceptance: field term + whole-voxel White Bear reaction work
            float dPhi = 0.0f;
            for (int i = 0; i < NS; ++i) if (dn[i]) dPhi += (float)dn[i] * phi[i][v];
            if (excl) {
                double xi[4]; xiAt(v, xi);
                double xn[4];
                for (int k = 0; k < 4; ++k) {
                    double d = 0.0;
                    for (int i = 0; i < NS; ++i) if (dn[i]) d += dn[i] * dxi[i + 1][k];
                    xn[k] = xi[k] + d;
                }
                dPhi += (float)(bFex(xn) - bFex(xi));
            }
            if (dPhi > 50.0f) dPhi = 50.0f;
            if (dPhi < -50.0f) dPhi = -50.0f;
            const float pi_r = std::fmin(1.0f, std::exp(-dPhi));
            if (urand(step, 5, (uint64_t)v, 0, P.seed) > pi_r) continue;
            // apply: the first reactant is RELABELLED to the first product, so a fusion's
            // survivor keeps its slot exactly as the GPU kernel does; any second reactant is
            // deleted and any second product appended.
            int need = (R.p1 >= 0 ? 1 : 0) - (R.r1 >= 0 ? 1 : 0);
            if (need > 0 && cnt[v] + need > CAP) { ++overflow; continue; }
            bool doneR0 = false, doneR1 = (R.r1 < 0);
            uint8_t tmp[CAP]; int w = 0;
            for (int i = 0; i < cnt[v]; ++i) {
                uint8_t sp = s[i];
                if (!doneR0 && sp == (uint8_t)(R.r0 + 1)) {
                    tmp[w++] = (uint8_t)(R.p0 + 1); doneR0 = true; continue;
                }
                if (!doneR1 && sp == (uint8_t)(R.r1 + 1)) { doneR1 = true; continue; }
                tmp[w++] = sp;
            }
            if (!doneR0 || !doneR1) continue;          // counts changed under us: skip
            if (R.p1 >= 0) tmp[w++] = (uint8_t)(R.p1 + 1);
            std::memcpy(s, tmp, w);
            std::memset(s + w, 0, CAP - w);
            cnt[v] = (uint8_t)w;
        }
    }

    void step(uint64_t t) {
        diffusionPass(0, 4 * t + 0);
        diffusionPass(1, 4 * t + 1);
        diffusionPass(2, 4 * t + 2);
        reactionPass(4 * t + 3);
    }

    // Per-voxel total reaction propensity, needed for the reaction ceiling tau_rxn. Returns the
    // maximum over voxels and the mean, in 1/s.
    void propensityStats(double& amax, double& amean) const {
        amax = 0.0; double tot = 0.0;
        for (int v = 0; v < M; ++v) {
            const uint8_t* s = &lat[(size_t)v * CAP];
            int n[MAXS]; for (int i = 0; i < P.nspec; ++i) n[i] = 0;
            for (int i = 0; i < cnt[v]; ++i) ++n[s[i] - 1];
            double at = 0.0;
            for (int r = 0; r < P.nrxn; ++r) at += propensity(P.rxn[r], n);
            if (at > amax) amax = at;
            tot += at;
        }
        amean = tot / (double)M;
    }
    void countsBySpecies(long* out) const {
        for (int i = 0; i < P.nspec; ++i) out[i] = 0;
        for (int v = 0; v < M; ++v) {
            const uint8_t* s = &lat[(size_t)v * CAP];
            for (int i = 0; i < cnt[v]; ++i) ++out[s[i] - 1];
        }
    }
    // Occupancy histogram over voxels, for the extreme-value estimate of maxB.
    void occHistogram(std::vector<long>& hist) const {
        hist.assign(CAP + 1, 0);
        for (int v = 0; v < M; ++v) ++hist[cnt[v]];
    }
};

static void parseRxn(const std::string& spec, Rxn& R) {
    // <kind>:<r0>[,<r1>]/<p0>[,<p1>]@<rate>
    size_t colon = spec.find(':'), gt = spec.find('/'), at = spec.find('@');
    if (colon == std::string::npos || gt == std::string::npos || at == std::string::npos) {
        fprintf(stderr, "bad --rxn '%s'\n", spec.c_str()); exit(2);
    }
    std::string kind = spec.substr(0, colon);
    std::string lhs = spec.substr(colon + 1, gt - colon - 1);
    std::string rhs = spec.substr(gt + 1, at - gt - 1);
    R.rate = atof(spec.substr(at + 1).c_str());
    R.r0 = R.r1 = R.p0 = R.p1 = -1;
    if (sscanf(lhs.c_str(), "%d,%d", &R.r0, &R.r1) < 2) R.r1 = -1;
    if (sscanf(rhs.c_str(), "%d,%d", &R.p0, &R.p1) < 2) R.p1 = -1;
    if (kind == "fusion") R.kind = FUSION;
    else if (kind == "fission") R.kind = FISSION;
    else if (kind == "conversion") R.kind = CONVERSION;
    else { fprintf(stderr, "bad kind '%s'\n", kind.c_str()); exit(2); }
}

static void parseList(const char* s, double* out, int n) {
    std::string t(s); size_t p = 0;
    for (int i = 0; i < n; ++i) {
        size_t c = t.find(',', p);
        out[i] = atof(t.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str());
        if (c == std::string::npos) break;
        p = c + 1;
    }
}

int main(int argc, char** argv) {
    Params P;
    long steps1 = 200, steps2 = 1000;
    std::string out = "adv_cpu.json";
    std::vector<std::string> rxnSpecs;
    std::string seedCounts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--dims") sscanf(next().c_str(), "%d,%d,%d", &P.NX, &P.NY, &P.NZ);
        else if (a == "--nspec") P.nspec = atoi(next().c_str());
        else if (a == "--sigma") parseList(next().c_str(), P.sigma, MAXS);
        else if (a == "--gamma") parseList(next().c_str(), P.gamma, MAXS);
        else if (a == "--seed-counts") seedCounts = next();
        else if (a == "--rxn") rxnSpecs.push_back(next());
        else if (a == "--h") P.h = atof(next().c_str());
        else if (a == "--D") P.D = atof(next().c_str());
        else if (a == "--dt") P.dt = atof(next().c_str());
        else if (a == "--wavelength") P.wavelength = atof(next().c_str());
        else if (a == "--no-exclusion") P.exclusion = false;
        else if (a == "--no-stokes") P.stokes = false;
        else if (a == "--seed") P.seed = strtoull(next().c_str(), 0, 10);
        else if (a == "--steps1") steps1 = atol(next().c_str());
        else if (a == "--steps2") steps2 = atol(next().c_str());
        else if (a == "--out") out = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (P.nspec > MAXS) { fprintf(stderr, "nspec > %d\n", MAXS); return 2; }
    P.nrxn = (int)rxnSpecs.size();
    if (P.nrxn > MAXR) { fprintf(stderr, "too many --rxn\n"); return 2; }
    for (int r = 0; r < P.nrxn; ++r) parseRxn(rxnSpecs[r], P.rxn[r]);
    {   // "0:1449,1:1449"
        std::string t = seedCounts; size_t p = 0;
        while (p < t.size()) {
            size_t c = t.find(',', p);
            std::string item = t.substr(p, c == std::string::npos ? std::string::npos : c - p);
            int sp; long n;
            if (sscanf(item.c_str(), "%d:%ld", &sp, &n) == 2 && sp >= 0 && sp < MAXS)
                P.seedCount[sp] = n;
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }

    Sim S; S.P = P; S.init();
    long n0total = 0;
    for (int s = 0; s < P.nspec; ++s)
        if (P.seedCount[s]) { S.place(s + 1, P.seedCount[s], 0x1000ULL * (s + 1)); n0total += P.seedCount[s]; }

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    for (long t = 0; t < steps1; ++t) S.step((uint64_t)t);
    auto t1 = clk::now();
    for (long t = steps1; t < steps2; ++t) S.step((uint64_t)t);
    auto t2 = clk::now();
    double w1 = std::chrono::duration<double>(t1 - t0).count();
    double w2 = std::chrono::duration<double>(t2 - t1).count();
    double per_step = (steps2 > steps1) ? w2 / (double)(steps2 - steps1) : w1 / (double)steps1;

    double amax = 0.0, amean = 0.0; S.propensityStats(amax, amean);
    long cbs[MAXS]; S.countsBySpecies(cbs);
    std::vector<long> hist; S.occHistogram(hist);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    FILE* f = fopen(out.c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    fprintf(f, "{\"engine\":\"lat_cpu\",\"dims\":[%d,%d,%d],\"M\":%d,\"nspec\":%d,\"n0\":%ld,"
               "\"threads\":%d,\"exclusion\":%d,\"stokes\":%d,\"dt_s\":%g,\"steps1\":%ld,\"steps2\":%ld,"
               "\"wall1_s\":%g,\"wall2_s\":%g,\"per_step_s\":%g,\"a_max_per_s\":%g,"
               "\"a_mean_per_s\":%g,\"overflow\":%ld,\"seed\":%llu,\"counts\":[",
            P.NX, P.NY, P.NZ, S.M, P.nspec, n0total, threads, (int)P.exclusion, (int)P.stokes, P.dt,
            steps1, steps2, w1, w2, per_step, amax, amean, S.overflow,
            (unsigned long long)P.seed);
    for (int s = 0; s < P.nspec; ++s) fprintf(f, "%s%ld", s ? "," : "", cbs[s]);
    fprintf(f, "],\"occ_hist\":[");
    int hmax = CAP; while (hmax > 0 && hist[hmax] == 0) --hmax;
    for (int i = 0; i <= hmax; ++i) fprintf(f, "%s%ld", i ? "," : "", hist[i]);
    fprintf(f, "]}\n");
    fclose(f);
    printf("ADV_CPU dims=%d,%d,%d nspec=%d n0=%ld threads=%d per_step=%.6g s a_max=%.4g /s "
           "overflow=%ld\n", P.NX, P.NY, P.NZ, P.nspec, n0total, threads, per_step, amax,
           S.overflow);
    return 0;
}
