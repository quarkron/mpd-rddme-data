// structure.cpp -- the overlay and the complex registry: seeding bodies and chains, the L26 bond
// encoding the overlay uses (struct_complex_pod.h), and the smear quantisation of a centre.
#include "sim.h"
#include "structure.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int structQuant(double com) { return (int)std::floor(com * (double)SMEAR_COM_Q + 0.5); }

int dirIndex(int dx, int dy, int dz) {
    const int raw = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1);
    return raw < 13 ? raw : raw - 1;
}

void dirDecode(int d, int& dx, int& dy, int& dz) {
    const int raw = d < 13 ? d : d + 1;
    dx = raw % 3 - 1; dy = (raw / 3) % 3 - 1; dz = raw / 9 - 1;
}

// Rewrite the overlay of complex slot k from its path: ids, +ward bonds, roles, fill 1.
void writeChainOverlay(Sim& s, int k) {
    const Complex& c = s.cx[k];
    const int n = (int)c.vox.size();
    for (int i = 0; i < n; ++i) {
        const int v = c.vox[i];
        s.l1[v] = (uint32_t)(k + 1);
        if (i + 1 < n) {
            int ax, ay, az, bx, by, bz;
            s.coords(v, ax, ay, az); s.coords(c.vox[i + 1], bx, by, bz);
            s.l2[v] = (uint8_t)dirIndex(bx - ax, by - ay, bz - az);
        } else {
            s.l2[v] = DIR_NONE;
        }
        const int role = (i == 0) ? ROLE_NEGTIP : (i == n - 1) ? ROLE_POSTIP : ROLE_BODY;
        s.l3[v] = (uint8_t)((role << 4) | 1);
    }
}

void clearVoxel(Sim& s, int v) { s.l1[v] = 0; s.l2[v] = DIR_NONE; s.l3[v] = 0; }

void chainCom(Sim& s, int k) {
    Complex& c = s.cx[k];
    double a[3] = {0, 0, 0};
    for (int v : c.vox) { int x, y, z; s.coords(v, x, y, z); a[0] += x; a[1] += y; a[2] += z; }
    const double n = (double)c.vox.size();
    for (int i = 0; i < 3; ++i) c.com[i] = a[i] / n;
}

void seedStructures(Sim& s) {
    const Scene& S = s.S;
    const int nb = (int)(S.bodyCom.size() / 3);
    for (int b = 0; b < nb; ++b) {
        const int k = s.newSlot();
        Complex& c = s.cx[k];
        c.active = true; c.body = true; c.species = S.bodySpecies;
        for (int a = 0; a < 3; ++a) c.com[a] = S.bodyCom[3 * b + a];
        const int v = s.vid((int)std::floor(c.com[0] + 0.5), (int)std::floor(c.com[1] + 0.5), (int)std::floor(c.com[2] + 0.5));
        if (s.l1[v]) { fprintf(stderr, "[seed] body %d lands on an occupied overlay voxel %d\n", b, v); exit(3); }
        c.vox.assign(1, v);
        s.l1[v] = (uint32_t)(k + 1); s.l2[v] = DIR_NONE; s.l3[v] = (uint8_t)((ROLE_BODY << 4) | 1);
    }
    size_t off = 0;
    for (size_t ch = 0; ch < S.chainLen.size(); ++ch) {
        const int n = S.chainLen[ch];
        const int k = s.newSlot();
        Complex& c = s.cx[k];
        c.active = true; c.body = false; c.species = S.fillSpecies;
        c.vox.resize(n);
        for (int i = 0; i < n; ++i, ++off) {
            const int v = s.vid(S.chainVox[3 * off], S.chainVox[3 * off + 1], S.chainVox[3 * off + 2]);
            if (s.l1[v]) { fprintf(stderr, "[seed] chain %zu subunit %d lands on an occupied voxel\n", ch, i); exit(3); }
            c.vox[i] = v;
            s.l1[v] = (uint32_t)(k + 1);
        }
        writeChainOverlay(s, k);
        chainCom(s, k);
    }
}
