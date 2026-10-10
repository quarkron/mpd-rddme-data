// main.cpp -- driver of the single-core port. Loads a scene directory, seeds it, runs steps1 untimed
// and times [steps1, steps2), recording validation observables every frameEvery steps when asked.
//
//   usage: lmcpu <scene_dir> [--steps1 N] [--steps2 N] [--frame-every N] [--seed N] [--out PREFIX]
//                [--nuc-count-first 0|1]   (1: count monomers before the nucleation core scan; same outcomes)
//
// Writes <out>.json (cost and counts) and, with frames, <out>_occ.f64 (per frame and type: mean and
// variance of the per-voxel count), <out>_com.f64 (per frame: every body's centre) and, for chains,
// <out>_chains.i64 (per frame: chains, bound subunits, length histogram 0..64, then the mean R_ee^2
// and R_g^2 over chains with n > 1 as two doubles).
#include "sim.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void seedStructures(Sim& s);

static void recordFrame(const Sim& s, FILE* focc, FILE* fcom, FILE* fch) {
    const int nt = s.S.nt;
    std::vector<double> sum(nt + 1, 0.0), sum2(nt + 1, 0.0);
    std::vector<int> n(nt + 1);
    for (int v = 0; v < s.M; ++v) {
        std::fill(n.begin(), n.end(), 0);
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        for (int i = 0; i < s.cnt[v]; ++i) ++n[p[i]];
        for (int t = 1; t <= nt; ++t) { sum[t] += n[t]; sum2[t] += (double)n[t] * n[t]; }
    }
    for (int t = 1; t <= nt; ++t) {
        const double m = sum[t] / s.M, var = sum2[t] / s.M - m * m;
        fwrite(&m, sizeof m, 1, focc); fwrite(&var, sizeof var, 1, focc);
    }
    for (const Complex& c : s.cx)
        if (c.active && c.body) fwrite(c.com, sizeof(double), 3, fcom);
    if (fch) {
        std::vector<long long> row(2 + 65, 0);
        for (const Complex& c : s.cx) {
            if (!c.active || c.body) continue;
            ++row[0]; row[1] += (long long)c.vox.size();
            if (c.vox.size() <= 64) ++row[2 + c.vox.size()];
        }
        fwrite(row.data(), sizeof(long long), row.size(), fch);
        double ree = 0.0, rg = 0.0; long nsz = 0;       // gradient_hook: means over chains with n > 1
        for (const Complex& c : s.cx) {
            if (!c.active || c.body || c.vox.size() < 2) continue;
            double m[3] = {0, 0, 0}, a[3], b[3];
            std::vector<double> p(3 * c.vox.size());
            for (size_t i = 0; i < c.vox.size(); ++i) {
                int x, y, z; s.coords(c.vox[i], x, y, z);
                p[3 * i] = x; p[3 * i + 1] = y; p[3 * i + 2] = z; m[0] += x; m[1] += y; m[2] += z;
            }
            const double n = (double)c.vox.size(); double g = 0.0;
            for (int k = 0; k < 3; ++k) { m[k] /= n; a[k] = p[k]; b[k] = p[3 * (c.vox.size() - 1) + k]; }
            for (size_t i = 0; i < c.vox.size(); ++i)
                for (int k = 0; k < 3; ++k) g += (p[3 * i + k] - m[k]) * (p[3 * i + k] - m[k]);
            ree += (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2]);
            rg += g / n; ++nsz;
        }
        const double sz[2] = {nsz ? ree / nsz : 0.0, nsz ? rg / nsz : 0.0};
        fwrite(sz, sizeof(double), 2, fch);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: lmcpu <scene_dir> [--steps1 N] [--steps2 N] [--frame-every N] [--seed N] [--out P]\n"); return 2; }
    Scene S = loadScene(argv[1]);
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); return 2; }
        const char* v = argv[++i];
        if (a == "--steps1") S.steps1 = atol(v);
        else if (a == "--steps2") S.steps2 = atol(v);
        else if (a == "--frame-every") S.frameEvery = atol(v);
        else if (a == "--seed") S.seed = strtoull(v, 0, 10);
        else if (a == "--out") S.outPrefix = v;
        else if (a == "--nuc-count-first") S.nucCountFirst = atoi(v) != 0;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    Sim s(S);
    seedStructures(s);
    long n0 = 0;
    for (int t = 1; t <= S.nt; ++t) if (S.count[t]) { s.placeParticles(t, S.count[t]); n0 += S.count[t]; }

    FILE *focc = nullptr, *fcom = nullptr, *fch = nullptr;
    if (S.frameEvery > 0) {
        focc = fopen((S.outPrefix + "_occ.f64").c_str(), "wb");
        fcom = fopen((S.outPrefix + "_com.f64").c_str(), "wb");
        if (S.chains) fch = fopen((S.outPrefix + "_chains.i64").c_str(), "wb");
        if (!focc || !fcom || (S.chains && !fch)) { fprintf(stderr, "cannot open frame files at %s\n", S.outPrefix.c_str()); return 1; }
        recordFrame(s, focc, fcom, fch);
    }
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now(), t1 = t0;
    long frames = S.frameEvery > 0 ? 1 : 0;
    for (long t = 0; t < S.steps2; ++t) {
        if (t == S.steps1) t1 = clk::now();
        s.step();
        if (S.frameEvery > 0 && (t + 1) % S.frameEvery == 0) { recordFrame(s, focc, fcom, fch); ++frames; }
    }
    auto t2 = clk::now();
    if (S.steps1 >= S.steps2) t1 = t0;
    const double w1 = std::chrono::duration<double>(t1 - t0).count();
    const double w2 = std::chrono::duration<double>(t2 - t1).count();
    const long nTimed = S.steps2 - (S.steps1 < S.steps2 ? S.steps1 : 0);
    const double perStep = w2 / (double)(nTimed > 0 ? nTimed : 1);
    if (focc) fclose(focc);
    if (fcom) fclose(fcom);
    if (fch) fclose(fch);

    long counts[MAXT] = {0};
    for (int v = 0; v < s.M; ++v) for (int i = 0; i < s.cnt[v]; ++i) ++counts[s.lat[(size_t)v * CAP + i]];
    long nb = 0, nch = 0, nbound = 0;
    for (const Complex& c : s.cx) { if (!c.active) continue; if (c.body) ++nb; else { ++nch; nbound += (long)c.vox.size(); } }
    FILE* f = fopen((S.outPrefix + ".json").c_str(), "w");
    if (!f) { fprintf(stderr, "cannot write %s.json\n", S.outPrefix.c_str()); return 1; }
    fprintf(f, "{\"engine\":\"lat_cpu_port\",\"scene\":\"%s\",\"dims\":[%d,%d,%d],\"n0\":%ld,\"tau_s\":%g,"
               "\"steps1\":%ld,\"steps2\":%ld,\"wall1_s\":%.6f,\"wall2_s\":%.6f,\"per_step_s\":%.9g,"
               "\"overflow\":%ld,\"frames\":%ld,\"frame_every\":%ld,\"seed\":%llu,\"bodies\":%ld,"
               "\"chains\":%ld,\"bound\":%ld,\"nuc_count_first\":%d,\"counts\":[",
            argv[1], S.NX, S.NY, S.NZ, n0, S.tau, S.steps1, S.steps2, w1, w2, perStep, s.overflow, frames,
            S.frameEvery, (unsigned long long)S.seed, nb, nch, nbound, (int)S.nucCountFirst);
    for (int t = 1; t <= S.nt; ++t) fprintf(f, "%s%ld", t > 1 ? "," : "", counts[t]);
    std::vector<long> hist(CAP + 1, 0);
    for (int v = 0; v < s.M; ++v) ++hist[s.cnt[v]];
    int hmax = CAP; while (hmax > 0 && !hist[hmax]) --hmax;
    fprintf(f, "],\"occ_hist\":[");
    for (int i = 0; i <= hmax; ++i) fprintf(f, "%s%ld", i ? "," : "", hist[i]);
    fprintf(f, "]}\n");
    fclose(f);
    printf("LMCPU %s per_step=%.6g s (timed %ld steps) overflow=%ld bodies=%ld chains=%ld bound=%ld\n",
           argv[1], perStep, nTimed, s.overflow, nb, nch, nbound);
    return 0;
}
