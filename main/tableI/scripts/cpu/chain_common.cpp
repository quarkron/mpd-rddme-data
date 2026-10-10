// chain_common.cpp -- see chain_common.h. Device sources: byte_structure_drift_dev.cu (structDirCos,
// structJointBend, structBondStretch, structOctApply, structVexInsertDelta, structExtSmearInsRaw),
// struct_chain_friction_dev.cuh, struct_formation_excess_dev.cuh, struct_bead_body_dev.cuh.
#include "chain_common.h"
#include "wb.h"
#include <cmath>

float dirCos(int a, int b) {
    int ax, ay, az, bx, by, bz;
    l26Dir(a, ax, ay, az); l26Dir(b, bx, by, bz);
    const float la = sqrtf((float)(ax * ax + ay * ay + az * az)), lb = sqrtf((float)(bx * bx + by * by + bz * bz));
    return (float)(ax * bx + ay * by + az * bz) / (la * lb);
}

float jointBend(int dirIn, int dirOut, float kappa) { return kappa * (1.0f - dirCos(dirIn, dirOut)); }

float bondStretch(const Scene& S, int d) {
    if (S.kStretch == 0.0) return 0.0f;
    int dx, dy, dz; l26Dir(d, dx, dy, dz);
    const float dl = sqrtf((float)(dx * dx + dy * dy + dz * dz)) - (float)S.bond0;
    return 0.5f * (float)S.kStretch * dl * dl;
}

static const int OCT_PERM[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
static const int OCT_EVEN[6] = {1, 0, 0, 1, 1, 0};
static const int OCT_SE[4][3] = {{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
static const int OCT_SO[4][3] = {{-1, 1, 1}, {1, -1, 1}, {1, 1, -1}, {-1, -1, -1}};

void octApply(int g, int vx, int vy, int vz, int& ox, int& oy, int& oz) {
    const int gp = g >> 2, gs = g & 3;
    const int* sg = OCT_EVEN[gp] ? OCT_SE[gs] : OCT_SO[gs];
    const int c[3] = {vx, vy, vz};
    ox = sg[0] * c[OCT_PERM[gp][0]]; oy = sg[1] * c[OCT_PERM[gp][1]]; oz = sg[2] * c[OCT_PERM[gp][2]];
}

void octApplyReal(int g, double vx, double vy, double vz, double& ox, double& oy, double& oz) {
    const int gp = g >> 2, gs = g & 3;
    const int* sg = OCT_EVEN[gp] ? OCT_SE[gs] : OCT_SO[gs];
    const double c[3] = {vx, vy, vz};
    ox = sg[0] * c[OCT_PERM[gp][0]]; oy = sg[1] * c[OCT_PERM[gp][1]]; oz = sg[2] * c[OCT_PERM[gp][2]];
}

int octRotateDir(int g, int d) {
    if (d == DIR_NONE) return DIR_NONE;
    int dx, dy, dz; l26Dir(d, dx, dy, dz);
    int ox, oy, oz; octApply(g, dx, dy, dz, ox, oy, oz);
    const int raw = (ox + 1) + 3 * (oy + 1) + 9 * (oz + 1);
    return raw < 13 ? raw : raw - 1;
}

static double shellSum(const Sim& s, int x0) {
    double acc = 0.0; int nIn = 0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy && !dz) continue;
        const int v = s.snb(x0, dx, dy, dz);
        if (v < 0) continue;
        const uint8_t* p = &s.lat[(size_t)v * CAP];
        for (int i = 0; i < s.cnt[v]; ++i) {
            const Species& sp = s.S.sp[p[i]];
            if (sp.fricChain > 0.0f) acc += (double)sp.fricChain * sp.dxi[3];
        }
        ++nIn;
    }
    return nIn ? acc / nIn : 0.0;
}

bool chainFrictionThin(Sim& s, const int* oldv, const int* neww, int n) {
    double F = 0.0;
    for (int k = 0; k < n; ++k) {
        F += shellSum(s, oldv[k]);
        F += shellSum(s, neww[k] == oldv[k] ? oldv[k] : neww[k]);
    }
    F /= (double)(2 * (n > 0 ? n : 1));
    const double m = 1.0 / (1.0 + F);
    if (m >= 1.0) return false;
    return s.rng.ud() >= m;
}

// The bath the chain's moves still feel: types NOT rate-coupled (fricChain == 0), plus other complexes' fill.
static void chainBathXi(const Sim& s, int cid, int v, double x[4]) {
    x[0] = x[1] = x[2] = x[3] = 0.0;
    const uint8_t* p = &s.lat[(size_t)v * CAP];
    for (int i = 0; i < s.cnt[v]; ++i) {
        const Species& sp = s.S.sp[p[i]];
        if (sp.fricChain > 0.0f) continue;
        for (int k = 0; k < 4; ++k) x[k] += sp.dxi[k];
    }
    int fill = s.fillAt(v);
    if ((int)s.l1[v] == cid) fill = 0;
    if (fill > 0) for (int k = 0; k < 4; ++k) x[k] += fill * s.S.sp[s.S.fillSpecies].dxi[k];
}

static double chainVexInsert(const Sim& s, int cid, int v) {
    if (s.S.lambda == 0.0f) return 0.0;
    double x[4]; chainBathXi(s, cid, v, x);
    const double* d = s.S.sp[s.S.fillSpecies].dxi;
    return s.S.lambda * (bFexWBd(s.S.V, x[0] + d[0], x[1] + d[1], x[2] + d[2], x[3] + d[3]) - bFexWBd(s.S.V, x[0], x[1], x[2], x[3]));
}

static double extSmearInsRaw(const Sim& s, int t, int v) {
    const double* d = s.S.sp[t].dxi;
    const size_t M = (size_t)s.M;
    const double* f = s.smearBase.data();          // bead-body armed and a baseline published: the baseline
    const double s0 = f[v], s1 = f[M + v], s2 = f[2 * M + v], s3 = f[3 * M + v];
    return bFexWBd(s.S.V, s0 + d[0], s1 + d[1], s2 + d[2], s3 + d[3]) - bFexWBd(s.S.V, s0, s1, s2, s3);
}

double vexInsertDelta(const Sim& s, int cid, int species, int src, int dst) {
    double d = chainVexInsert(s, cid, dst) - chainVexInsert(s, cid, src);
    if (species != s.S.bodySpecies && src != dst && s.S.lambda != 0.0f)
        d += s.S.lambda * (extSmearInsRaw(s, species, dst) - extSmearInsRaw(s, species, src));
    return d;
}

double formationExcess(const Sim& s, int cid, int species, int v, int m) {
    const double V = s.S.V;
    double x[4]; chainBathXi(s, cid, v, x);
    const double* dm = s.S.sp[m].dxi;
    for (int k = 0; k < 4; ++k) { x[k] -= dm[k]; if (x[k] < 0.0) x[k] = 0.0; }
    const double* df = s.S.sp[s.S.fillSpecies].dxi;
    const double vex = s.S.lambda * (bFexWBd(V, x[0] + df[0], x[1] + df[1], x[2] + df[2], x[3] + df[3])
                                     - bFexWBd(V, x[0] + dm[0], x[1] + dm[1], x[2] + dm[2], x[3] + dm[3]));
    const double smear = s.S.lambda * (extSmearInsRaw(s, species, v) - extSmearInsRaw(s, m, v));
    return vex + smear;                          // the one-body field part is zero in this scene
}

bool beadBodyBlocksMove(const Sim& s, int cid, const int* dest, int n) {
    for (int i = 0; i < n; ++i) if (subunitHitsBody(s, cid, dest[i])) return true;
    return false;
}
