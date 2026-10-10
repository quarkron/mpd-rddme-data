// lattice.cpp -- construction, particle bookkeeping on the byte lattice, and the step in the GPU's
// order (DriftMpdRdmeSolver::runTimestep, then launchStructurePasses): X, Y, Z bath passes; smear;
// chain collective hop; body sub-voxel step; interior moves; tips; nucleation (the SI move set has no
// rigid rotation).
#include "sim.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

Sim::Sim(const Scene& sc) : S(sc), rng(sc.seed) {
    M = S.NX * S.NY * S.NZ;
    lat.assign((size_t)M * CAP, 0); nlat = lat;
    cnt.assign(M, 0); ncnt = cnt;
    choice.assign((size_t)M * CAP, 0);
    xi.assign((size_t)M * 4, 0.0f);
    Fv.assign(M, 0.0f); wv.assign(M, 0.0f);
    l1.assign(M, 0); l2.assign(M, DIR_NONE); l3.assign(M, 0);
    smear.assign((size_t)4 * M, 0.0); smearBase = smear;
}

bool Sim::insertParticle(int v, int type) {
    if (cnt[v] >= CAP) return false;
    lat[(size_t)v * CAP + cnt[v]++] = (uint8_t)type;
    return true;
}

bool Sim::removeParticle(int v, int type) {
    uint8_t* s = &lat[(size_t)v * CAP];
    for (int i = 0; i < cnt[v]; ++i)
        if (s[i] == type) { s[i] = s[--cnt[v]]; s[cnt[v]] = 0; return true; }
    return false;
}

int Sim::countType(int v, int type) const {
    const uint8_t* s = &lat[(size_t)v * CAP];
    int n = 0;
    for (int i = 0; i < cnt[v]; ++i) n += (s[i] == type);
    return n;
}

// Uniform over sites, as pyLM's addParticles places them; a full site draws again.
void Sim::placeParticles(int type, long n) {
    for (long i = 0; i < n; ++i) {
        for (int tries = 0;; ++tries) {
            if (tries > 1000) { fprintf(stderr, "[lattice] cannot place type %d: lattice full\n", type); exit(3); }
            if (insertParticle((int)rng.below((uint32_t)M), type)) break;
        }
    }
}

int Sim::newSlot() {
    if (!freeSlots.empty()) { int k = freeSlots.back(); freeSlots.pop_back(); return k; }
    cx.emplace_back();
    frozen.push_back(-1); frozen.push_back(-1); frozen.push_back(-1);
    return (int)cx.size() - 1;
}

// LMCPU_PROFILE=1 in the environment times each operator (chrono around each call; a few percent of
// overhead, so never on for a timed cost run) and prints the split at exit.
static const char* OPN[10] = {"bath x", "bath y", "bath z", "smear", "chain hop", "bodies", "(unused)", "interior", "tips", "nucleation"};
static double opT[10];
static bool profiling() { static int p = -1; if (p < 0) { const char* e = getenv("LMCPU_PROFILE"); p = (e && *e == '1'); } return p; }
static void printProfile() {
    double tot = 0; for (double t : opT) tot += t;
    for (int i = 0; i < 10; ++i) if (opT[i] > 0) fprintf(stderr, "[profile] %-16s %8.3f s  %5.1f %%\n", OPN[i], opT[i], 100 * opT[i] / tot);
}

void Sim::step() {
    using clk = std::chrono::steady_clock;
    const bool prof = profiling();
    static bool registered = false;
    if (prof && !registered) { atexit(printProfile); registered = true; }
    auto t = clk::now();
    auto lap = [&](int i) { if (!prof) return; auto n = clk::now(); opT[i] += std::chrono::duration<double>(n - t).count(); t = n; };
    // A statics scene (no bath particles, no bodies, no chemistry) has nothing for the bath passes or
    // the smear to act on; skipping them lets its box be large enough that a chain never meets a wall.
    // Never active in a timed scene: any particle, body or rate switches it off.
    static int bare = -1;
    if (bare < 0) {
        long n = 0; for (int t = 1; t <= S.nt; ++t) n += S.count[t];
        bare = (n == 0 && S.bodyCom.empty() && S.kNuc == 0.0 && S.kPoly == 0.0 && S.kDepoly == 0.0) ? 1 : 0;
    }
    if (!bare) {
        bathPass(*this, 0); lap(0);
        bathPass(*this, 1); lap(1);
        bathPass(*this, 2); lap(2);
        smearStep(*this); lap(3);
    }
    if (S.chains) chainHopAll(*this);        // the hop kernel (chains) runs before the body cell kernel
    lap(4);
    bodyStepAll(*this); lap(5);
    if (S.chains) {
        if (S.interiorAttempts > 0) chainInteriorAll(*this);
        lap(7);
        chainTipAll(*this); lap(8);
        chainNucleation(*this); lap(9);
    }
    ++stepIndex;
}
