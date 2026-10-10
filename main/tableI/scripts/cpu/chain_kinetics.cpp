// chain_kinetics.cpp -- the chain chemistry: the tip kernel (mpd_structure_tip_kernel with structTipAdd
// and structSampleGrowthDirUnified) and order-1 nucleation (mpd_structure_nucleation_kernel).
//
// Per chain, in the device's order: a one-subunit chain dissolves at 1 - exp(-k_- tau); a longer one
// retracts its - tip at the same rate (bond energy 0, end energy 0); a chain of four or more sheds
// its + tip at the same rate; then both tips try to grow. A tip's growth weights run over its 26
// neighbours: an empty, unstructured neighbour holding z monomers, outside every body's core, weighs
// z exp(-kappa (1 - cos) - stretch - formation swap), with the angle measured against the tip's
// reference heading; the tip fires at 1 - exp(-k_+ sum w tau) into a direction drawn from the weights.
// Nucleation visits every unstructured voxel, rejects it inside a body's core (checked BEFORE the
// monomer count, as the device does), and seeds a one-subunit chain at 1 - exp(-k_nuc z tau).
#include "chain_common.h"
#include "structure.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void releaseMonomer(Sim& s, int v) {
    if (!s.insertParticle(v, s.S.monomerSpecies)) {
        fprintf(stderr, "[chains] site %d full: cannot release a monomer (STRUCT_ERR_OCC_OVERFLOW)\n", v);
        exit(6);
    }
}

static void dissolve(Sim& s, int k) {
    Complex& c = s.cx[k];
    for (int v : c.vox) { releaseMonomer(s, v); clearVoxel(s, v); }
    c.vox.clear(); c.active = false;
    s.freeSlots.push_back(k);
}

// structSampleGrowthDirUnified + structTipAdd's new-voxel branch (fill capacity 1).
static bool tipAdd(Sim& s, int k, bool plusEnd) {
    Complex& c = s.cx[k];
    const Scene& S = s.S;
    const int cid = k + 1, n = (int)c.vox.size();
    const int tip = plusEnd ? c.vox.back() : c.vox.front();
    int refDir;
    if (plusEnd) refDir = (n >= 2) ? (int)s.l2[c.vox[n - 2]] : c.growthDir;
    else refDir = 25 - (int)s.l2[tip];                   // DIR_NONE at a lone voxel: decoded as the device does
    int cdx, cdy, cdz; l26Dir(refDir, cdx, cdy, cdz);
    const float clen = sqrtf((float)(cdx * cdx + cdy * cdy + cdz * cdz));
    const float udir = s.rng.uf();
    float w[26]; double wsum = 0.0;
    for (int d = 0; d < 26; ++d) {
        w[d] = 0.0f;
        int dx, dy, dz; l26Dir(d, dx, dy, dz);
        const int nb = s.snb(tip, dx, dy, dz);
        if (nb < 0 || s.fillAt(nb) >= 1 || s.l1[nb]) continue;
        const int z = s.countType(nb, S.monomerSpecies);
        if (z <= 0) continue;
        if (subunitHitsBody(s, cid, nb)) continue;
        const float dlen = sqrtf((float)(dx * dx + dy * dy + dz * dz));
        const float cth = (float)(dx * cdx + dy * cdy + dz * cdz) / (dlen * clen);
        const double eAng = (double)((float)S.kappa * (1.0f - cth) + bondStretch(S, d));
        const double env = formationExcess(s, cid, c.species, nb, S.monomerSpecies);
        w[d] = (float)(z * std::exp(-eAng - env));
        wsum += (double)w[d];
    }
    if (!(wsum > 0.0)) return false;
    int g = -1;
    { const double target = (double)udir * wsum; double acc = 0.0;
      for (int d = 0; d < 26; ++d) { acc += (double)w[d]; if (target < acc) { g = d; break; } } }
    if (g < 0) return false;
    const float Pp = 1.0f - (float)std::exp(-S.kPoly * wsum * S.tau);
    if (s.rng.uf() >= Pp) return false;
    int gx, gy, gz; l26Dir(g, gx, gy, gz);
    const int nb = s.snb(tip, gx, gy, gz);
    if (!s.removeParticle(nb, S.monomerSpecies)) return false;
    if (plusEnd) { c.vox.push_back(nb); c.growthDir = g; }
    else c.vox.insert(c.vox.begin(), nb);
    writeChainOverlay(s, k);
    return true;
}

void chainTipAll(Sim& s) {
    const Scene& S = s.S;
    const float Pdep = -expm1f(-(float)(S.kDepoly * S.tau));   // bond and end energies are 0: every channel at k_-
    const size_t nslots = s.cx.size();                          // newborns of this step do not react
    for (size_t k = 0; k < nslots; ++k) {
        Complex& c = s.cx[k];
        if (!c.active || c.body) continue;
        if (c.vox.size() == 1) {
            if (s.rng.uf() < Pdep) { dissolve(s, (int)k); continue; }
        } else {
            if (s.rng.uf() < Pdep) {                            // - tip retracts, its monomer stays behind
                const int v = c.vox.front();
                releaseMonomer(s, v); clearVoxel(s, v);
                c.vox.erase(c.vox.begin());
                writeChainOverlay(s, (int)k);
            }
        }
        if (c.vox.size() >= 4) {
            if (s.rng.uf() < Pdep) {                            // + tip sheds
                const int v = c.vox.back();
                releaseMonomer(s, v); clearVoxel(s, v);
                c.vox.pop_back();
                writeChainOverlay(s, (int)k);
            }
        }
        if (S.kPoly > 0.0) {
            tipAdd(s, (int)k, false);
            tipAdd(s, (int)k, true);
        }
        chainCom(s, (int)k);
    }
}

void chainNucleation(Sim& s) {
    const Scene& S = s.S;
    if (S.kNuc <= 0.0) return;
    for (int v = 0; v < s.M; ++v) {
        if (s.l1[v]) continue;
        int z = 0;
        if (S.nucCountFirst) {                 // single-core ordering: the cheap test first, same outcome
            z = s.countType(v, S.monomerSpecies);
            if (z < 1 || subunitHitsBody(s, 0, v)) continue;
        } else {                               // the device's order: the core scan at every free voxel
            if (subunitHitsBody(s, 0, v)) continue;
            z = s.countType(v, S.monomerSpecies);
            if (z < 1) continue;
        }
        const float P = -expm1f(-(float)(S.kNuc * z * S.tau));
        if (s.rng.uf() >= P) continue;
        if (!s.removeParticle(v, S.monomerSpecies)) continue;
        const int k = s.newSlot();
        Complex& c = s.cx[k];
        c = Complex();
        c.active = true; c.body = false; c.species = S.fillSpecies; c.growthDir = 21;
        c.vox.assign(1, v);
        writeChainOverlay(s, k);
        chainCom(s, k);
    }
}
