// structure.h -- overlay helpers shared by the body and chain operators.
#ifndef LMCPU_STRUCTURE_H_
#define LMCPU_STRUCTURE_H_
#include "sim.h"

int dirIndex(int dx, int dy, int dz);            // L26 index of a unit offset, centre skipped
void dirDecode(int d, int& dx, int& dy, int& dz);
void writeChainOverlay(Sim& s, int k);           // ids, +ward bonds, roles and fill from the path
void clearVoxel(Sim& s, int v);
void chainCom(Sim& s, int k);                    // mean of the subunits' integer voxel coordinates
#endif
