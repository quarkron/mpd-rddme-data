// chain_common.h -- what the chain move and growth operators share: the L26 geometry, the bend and
// stretch energies, the octahedral group, the chain friction factor and the move-energy terms that
// the GPU evaluates for every moved subunit even where they vanish.
#ifndef LMCPU_CHAIN_COMMON_H_
#define LMCPU_CHAIN_COMMON_H_
#include "sim.h"

// structL26Dir with the device's integer arithmetic, including for an out-of-range index (a lone
// nucleus's minus end asks for 25 - DIR_NONE; the GPU decodes it as below, and so do we).
static inline void l26Dir(int d, int& dx, int& dy, int& dz) {
    const int raw = (d < 13) ? d : d + 1;
    dx = (raw % 3) - 1; dy = ((raw / 3) % 3) - 1; dz = (raw / 9) - 1;
}
float dirCos(int a, int b);                              // structDirCos
float jointBend(int dirIn, int dirOut, float kappa);     // kappa (1 - cos), mu = 0
float bondStretch(const Scene& S, int d);                // (k/2) (|d| - b0)^2
void octApply(int g, int vx, int vy, int vz, int& ox, int& oy, int& oz);
void octApplyReal(int g, double vx, double vy, double vz, double& ox, double& oy, double& oz);
int octRotateDir(int g, int d);

// Chain collective friction: m = 1 / (1 + mean over moved subunits and both ends of F_shell), with
// F_shell the mean over the in-bounds 26 neighbours of sum_t c_t dxi_t[3]. Returns true if thinned.
bool chainFrictionThin(Sim& s, const int* oldv, const int* neww, int n);
// The move-energy terms of one subunit moved src -> dst (structVexInsertDelta): the chain's White
// Bear insertion against the bath types that stay in its energy (none are, here: every sized mobile
// type is rate-coupled) plus the other complexes' fill, and the external smear read from the static
// baseline (ext-all). Zero in both scenes; evaluated because the device evaluates it.
double vexInsertDelta(const Sim& s, int cid, int species, int src, int dst);
// The formation swap of growth at v (structFormationExcess, insert = true, m = the monomer type).
double formationExcess(const Sim& s, int cid, int species, int v, int m);
// Would a subunit of complex cid at each of dest[0..n) overlap a body?
bool beadBodyBlocksMove(const Sim& s, int cid, const int* dest, int n);
#endif
