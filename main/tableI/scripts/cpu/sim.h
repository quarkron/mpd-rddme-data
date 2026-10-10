// sim.h -- the state one step reads and writes, and the operators that advance it. The bath lives on
// a periodic byte lattice (up to CAP particles per site, one type byte each); structures live on an
// overlay of the same grid, which is NOT periodic (structNeighbour returns NONE at a face).
#ifndef LMCPU_SIM_H_
#define LMCPU_SIM_H_
#include <cstdint>
#include <vector>
#include "rng.h"
#include "scene.h"

static const uint8_t DIR_NONE = 255;
// struct_complex_pod.h: an interior subunit carries ROLE_BODY too; the tips are marked
enum Role { ROLE_BODY = 0, ROLE_POSTIP = 1, ROLE_NEGTIP = 2, ROLE_JUNCTION = 3 };
static const int SMEAR_COM_Q = 1024;   // STRUCT_SMEAR_COM_Q

// One structure complex. A body is n = 1 of the depositing species with a continuous centre; a
// chain is an explicit voxel path from its - tip (vox.front()) to its + tip (vox.back()).
struct Complex {
    bool active = false;
    bool body = false;
    int species = 0;
    double com[3] = {0, 0, 0};
    std::vector<int> vox;
    int growthDir = 21;          // L26 index of (0,0,1), what the topology pass gives a lone voxel
};

struct Sim {
    Scene S;
    int M = 0;
    Rng rng;
    // bath
    std::vector<uint8_t> lat, cnt;       // lat[v*CAP + i] = type byte, cnt[v] = occupancy
    std::vector<uint8_t> nlat, ncnt;     // propagation targets, swapped per axis
    std::vector<uint8_t> choice;         // per slot: 0 stay, 1 minus, 2 plus
    std::vector<float> xi;               // 4 per voxel, recomputed per axis pass
    std::vector<float> Fv, wv;           // per voxel: F (WB or FV) and the free-volume weight
    long overflow = 0;
    // structure overlay
    std::vector<uint32_t> l1;            // complex id = slot + 1, 0 = none
    std::vector<uint8_t> l2;             // outgoing (+ward) bond direction, L26 index, DIR_NONE = none
    std::vector<uint8_t> l3;             // role << 4 | fill
    std::vector<Complex> cx;
    std::vector<int> freeSlots;
    // smear of the depositing bodies
    std::vector<double> smear;           // 4 * M, layout alpha * M + v
    std::vector<double> smearBase;       // static baseline (zeros in both scenes)
    std::vector<int> frozen;             // 3 per slot, quantised centre on record, -1 = none
    long stepIndex = 0;

    explicit Sim(const Scene& sc);
    inline int vid(int x, int y, int z) const { return (z * S.NY + y) * S.NX + x; }
    inline void coords(int v, int& x, int& y, int& z) const { x = v % S.NX; y = (v / S.NX) % S.NY; z = v / (S.NX * S.NY); }
    // non-periodic neighbour on the structure lattice; -1 off the lattice
    inline int snb(int v, int dx, int dy, int dz) const {
        int x, y, z; coords(v, x, y, z); x += dx; y += dy; z += dz;
        if (x < 0 || y < 0 || z < 0 || x >= S.NX || y >= S.NY || z >= S.NZ) return -1;
        return vid(x, y, z);
    }
    inline int fillAt(int v) const { return l3[v] & 0x0F; }
    void placeParticles(int type, long n);
    bool insertParticle(int v, int type);    // false if the site is full
    bool removeParticle(int v, int type);    // false if none of that type
    int countType(int v, int type) const;
    int newSlot();
    void step();                             // one full timestep in the GPU's order
};

// bath_hop.cpp
void bathPass(Sim& s, int axis);
// smear.cpp
void smearStep(Sim& s);
void smearFullRestamp(Sim& s);
// body.cpp
void bodyStepAll(Sim& s);
double frictionShellMean(const Sim& s, int v, bool chainCoef);
bool bodyBlocked(const Sim& s, int cid, double nx, double ny, double nz, int centreVox);
bool subunitHitsBody(const Sim& s, int cid, int v);
// structure.cpp
void seedStructures(Sim& s);
int structQuant(double com);
// chain_moves.cpp / chain_kinetics.cpp (demixing)
void chainHopAll(Sim& s);
void chainInteriorAll(Sim& s);
void chainTipAll(Sim& s);
void chainNucleation(Sim& s);
#endif
