// scene.h -- everything a run is determined by, as make_scene.py writes it. One struct, so the two
// scenes share one loader and one driver and differ only in data.
#ifndef LMCPU_SCENE_H_
#define LMCPU_SCENE_H_
#include <cstdint>
#include <string>
#include <vector>

static const int CAP = 64;       // MPD_LATTICE_MAX_OCCUPANCY of the canonical build
static const int MAXT = 16;      // byte types 1..MAXT-1

struct Species {
    float sigma = 0.0f;          // hard-sphere diameter, nm (0 = no excluded volume)
    float D = 0.0f;              // nm^2/s
    float q = 0.0f;              // per-direction hop number D tau / h^2
    double dxi[4] = {0, 0, 0, 0};  // (pi / 6V) sigma^k, the weighted-density increments of one particle
    float dxif[4] = {0, 0, 0, 0};
    float fric = 0.0f;           // body collisional-friction coefficient c_t (auto, StructureFriction.cu)
    float fricChain = 0.0f;      // chain collisional-friction coefficient (StructureChainFriction.cu)
};

struct Scene {
    // lattice
    int NX = 32, NY = 8, NZ = 8;
    double h = 10.0, V = 1000.0, tau = 50e-6;
    int nt = 3;                  // byte types 1..nt
    Species sp[MAXT];
    long count[MAXT] = {0};      // free particles of each type to place uniformly at t = 0
    // bath hop
    float lambda = 1.0f;         // White Bear prefactor; 0 = no hard core
    bool exactHop = true;        // EXACT (Poisson on the WPE rate) or LINEAR (legacy 0.5 split)
    bool freeVol = false;        // free-volume functional (ternary) vs smear folded into xi (demixing)
    bool reciprocal = true;      // the overlay fill enters the bath densities as fillSpecies
    int fillSpecies = 1;
    // smear of the depositing bodies
    int bodySpecies = 2;         // registry species tag of a depositing body
    int smearR = 1;              // stencil half-width
    std::vector<float> stencil;  // ((2R+1)^3, 4), x slowest
    double smearSigmaNm = 20.0;  // cbrt(6 V sum n3 / pi), as StructureSmear.cu derives it
    int smearIncremental = 0;    // 0 = full re-stamp every step; k = incremental, full re-stamp every k steps
    // body sub-voxel step
    int Q = 8;
    double bodyPdir = 0.005;     // per-direction hop probability per step, D_body tau / h^2
    bool bodyPair = true, beadBody = true, friction = true;
    double subunitSigmaNm = 5.0; // the structure handler's sigma (fill or subunit), for the bead-body contact
    // bodies and chains to seed
    std::vector<double> bodyCom;         // 3 per body, voxel units
    std::vector<int> chainLen;           // subunits per chain
    std::vector<int> chainVox;           // x,y,z per subunit, backbone order
    // chains (demixing): kinetics and mechanics
    bool chains = false;
    double kNuc = 0.0, kPoly = 0.0, kDepoly = 0.0;
    double kappa = 4.0, kStretch = 400.0, bond0 = 1.4142135623730951;
    int interiorAttempts = 0;
    bool nucCountFirst = false;          // count monomers before the bead-body scan (identical outcomes; not the device's order)
    int monomerSpecies = 1;              // byte type of a free monomer
    double chainD1 = 0.0;                // Rouse D1 tau / h^2 per direction for n = 1
    // run
    uint64_t seed = 1;
    long steps1 = 300, steps2 = 1500;    // timing window [steps1, steps2)
    long frameEvery = 0;                 // 0 = no observables; k = record every k steps
    std::string outPrefix = "run";
};

// Reads <dir>/params.txt and the binary arrays beside it; derives q, dxi and the friction
// coefficients. Exits with a message on anything malformed: no silent defaults for a missing key
// the scene declares it needs.
Scene loadScene(const std::string& dir);
#endif
