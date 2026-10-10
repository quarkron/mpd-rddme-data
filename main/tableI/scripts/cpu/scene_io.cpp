// scene_io.cpp -- load a scene directory: params.txt (one "key value..." per line) plus the binary
// arrays stencil.f32, bodies.f64, chain_len.i32, chain_vox.i32. Derives the per-species hop number,
// the weighted-density increments and both friction coefficient sets exactly as the GPU host does.
#include "scene.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

static void die(const std::string& m) { fprintf(stderr, "[scene] %s\n", m.c_str()); exit(2); }

template <class T>
static std::vector<T> readArray(const std::string& path, bool required) {
    std::vector<T> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { if (required) die("missing " + path); return out; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n % (long)sizeof(T)) die("size of " + path + " is not a multiple of its element");
    out.resize(n / sizeof(T));
    if (n && fread(out.data(), 1, n, f) != (size_t)n) die("short read " + path);
    fclose(f);
    return out;
}

Scene loadScene(const std::string& dir) {
    Scene S;
    std::ifstream in(dir + "/params.txt");
    if (!in) die("cannot open " + dir + "/params.txt");
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string k; ls >> k;
        if (k == "species") {                     // species <t> <sigma_nm> <D_nm2s> <count>
            int t; double sig, D; long n;
            if (!(ls >> t >> sig >> D >> n) || t < 1 || t >= MAXT) die("bad species line: " + line);
            S.sp[t].sigma = (float)sig; S.sp[t].D = (float)D; S.count[t] = n;
            continue;
        }
        std::string rest; std::getline(ls, rest);
        size_t p = rest.find_first_not_of(' ');
        kv[k] = (p == std::string::npos) ? "" : rest.substr(p);
    }
    auto has = [&](const char* k) { return kv.count(k) > 0; };
    auto D_ = [&](const char* k) { if (!has(k)) die(std::string("missing key ") + k); return atof(kv[k].c_str()); };
    auto I_ = [&](const char* k) { return (long)llround(D_(k)); };
    auto Dopt = [&](const char* k, double d) { return has(k) ? atof(kv[k].c_str()) : d; };

    S.NX = (int)I_("NX"); S.NY = (int)I_("NY"); S.NZ = (int)I_("NZ");
    S.h = D_("h"); S.V = S.h * S.h * S.h; S.tau = D_("tau"); S.nt = (int)I_("nt");
    S.lambda = (float)D_("lambda"); S.exactHop = I_("exactHop") != 0; S.freeVol = I_("freeVol") != 0;
    S.reciprocal = I_("reciprocal") != 0; S.fillSpecies = (int)I_("fillSpecies");
    S.bodySpecies = (int)I_("bodySpecies"); S.smearIncremental = (int)I_("smearIncremental");
    S.Q = (int)I_("Q"); S.bodyPdir = D_("bodyPdir");
    S.bodyPair = I_("bodyPair") != 0; S.beadBody = I_("beadBody") != 0; S.friction = I_("friction") != 0;
    S.subunitSigmaNm = D_("subunitSigmaNm");
    S.chains = I_("chains") != 0;
    if (S.chains) {
        S.kNuc = D_("kNuc"); S.kPoly = D_("kPoly"); S.kDepoly = D_("kDepoly");
        S.kappa = D_("kappa"); S.kStretch = D_("kStretch"); S.bond0 = D_("bond0");
        S.interiorAttempts = (int)I_("interiorAttempts"); S.monomerSpecies = (int)I_("monomerSpecies");
        S.chainD1 = D_("chainD1");
    }
    S.seed = (uint64_t)I_("seed"); S.steps1 = I_("steps1"); S.steps2 = I_("steps2");
    S.frameEvery = (long)Dopt("frameEvery", 0);
    S.outPrefix = has("outPrefix") ? kv["outPrefix"] : dir + "/run";
    if (S.nt < 1 || S.nt >= MAXT) die("nt out of range");

    // smear stencil and the body diameter it implies (StructureSmear.cu:25)
    S.stencil = readArray<float>(dir + "/stencil.f32", true);
    const int rows = (int)(S.stencil.size() / 4);
    int W = (int)llround(cbrt((double)rows));
    if (W * W * W != rows || (W % 2) != 1) die("stencil is not (2R+1)^3 x 4");
    S.smearR = (W - 1) / 2;
    double n3 = 0.0;
    for (int r = 0; r < rows; ++r) n3 += S.stencil[4 * r + 3];
    S.smearSigmaNm = cbrt(6.0 * n3 * S.V / M_PI);

    S.bodyCom = readArray<double>(dir + "/bodies.f64", false);
    if (S.bodyCom.size() % 3) die("bodies.f64 is not 3 doubles per body");
    S.chainLen = readArray<int>(dir + "/chain_len.i32", false);
    S.chainVox = readArray<int>(dir + "/chain_vox.i32", false);
    long nsub = 0; for (int n : S.chainLen) nsub += n;
    if ((long)S.chainVox.size() != 3 * nsub) die("chain_vox.i32 does not match chain_len.i32");

    // per-species tables
    for (int t = 1; t <= S.nt; ++t) {
        Species& s = S.sp[t];
        s.q = (float)(s.D * S.tau / (S.h * S.h));
        for (int k = 0; k < 4; ++k) {
            s.dxi[k] = (s.sigma > 0) ? (M_PI / (6.0 * S.V)) * pow((double)s.sigma, (double)k) : 0.0;
            s.dxif[k] = (float)s.dxi[k];
        }
    }
    // body collisional friction, auto coefficients (StructureFriction.cu:95-112)
    const double pT = S.bodyPdir;
    for (int t = 1; t <= S.nt; ++t) {
        Species& s = S.sp[t];
        if (!(s.sigma > 0) || !(s.q > 0)) continue;
        const double lam = s.sigma / S.smearSigmaNm;
        s.fric = (float)(0.5 * pT / (pT + s.q) * pow(1.0 + lam, 3) / pow(lam, 3));
    }
    // chain collisional friction, auto coefficients (StructureChainFriction.cu), fill species F
    const Species& F = S.sp[S.fillSpecies];
    if (F.sigma > 0 && F.q > 0)
        for (int t = 1; t <= S.nt; ++t) {
            Species& s = S.sp[t];
            if (!(s.sigma > 0) || !(s.q > 0)) continue;
            const double lam = s.sigma / F.sigma;
            s.fricChain = (float)(0.5 * F.q / (F.q + s.q) * pow(1.0 + lam, 3) / pow(lam, 3));
        }
    return S;
}
