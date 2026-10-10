// rng.h -- one sequential xoshiro256+ stream. The GPU draws a counter hash per decision because its
// decisions run in parallel; on one core a stream is the natural generator and costs less per draw
// than a hash, which is the fair comparison. Trajectories are not bit-comparable with the GPU.
#ifndef LMCPU_RNG_H_
#define LMCPU_RNG_H_
#include <cstdint>

struct Rng {
    uint64_t s[4];
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    explicit Rng(uint64_t seed = 1) { for (int i = 0; i < 4; ++i) s[i] = splitmix(seed); }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        const uint64_t r = s[0] + s[3], t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    // uniform in [0, 1), 24 bits, the resolution of the GPU's float draws
    inline float uf() { return (float)(next() >> 40) * (1.0f / 16777216.0f); }
    inline double ud() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    inline uint32_t below(uint32_t n) { return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32); }
};
#endif
