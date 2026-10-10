/*
 * Per-step operator order of DriftMpdRdmeSolver::runTimestep (change: randomized top-level operator splitting).
 *
 * A timestep applies its operators one after another -- the x, y and z drift-diffusion passes, the structure passes
 * (with the self-consistent field update attached to them), and the lattice reaction pass -- each reading the state the
 * previous one left (Lie-Trotter splitting). A FIXED order carries a first-order bias: the one-step error holds terms
 * (tau^2/2)[A, B] whose sign is set by the order, and they add up over a run. Drawing the order uniformly at random each
 * step gives every commutator both signs equally often, so the bias cancels in expectation and the averaged scheme is
 * second order, at the cost of one pass of each operator per step as before. Every operator leaves the equilibrium
 * distribution invariant on its own, so no order, fixed or random, moves the stationary state: what the order changes is
 * the dynamics (transients, relaxation times, driven steady states).
 *
 * The permutation is a Fisher-Yates shuffle keyed on (run seed, timestep, salt) through getTimestepSeed, so a run is
 * reproducible. The salts are disjoint from every substep salt the kernels use (0-3 drift/reaction, 7 RNG-origin shift),
 * and the shuffle consumes no kernel's random stream, so the kernels draw exactly what they draw in fixed order.
 *
 * The structure-pass smear refresh is a separate switch (setStructureSmearAfterMoves): it is not an operator but the
 * bookkeeping that recomputes the bodies' exclusion field from their positions, so it is never permuted; it only moves
 * from the head of the structure passes to after their move group.
 *
 * Both switches default OFF; OFF is byte-identical to the engine without them.
 */
#include "config.h"
#include "core/Print.h"
#include "rddme/DriftMpdRdmeSolver.h"

namespace lm {
namespace rdme {

namespace {
const uint32_t kStepOrderSaltBase = 100u;   // getTimestepSeed salts 101.. : disjoint from the kernels' 0-3 and 7
}

void DriftMpdRdmeSolver::setStepOrderRandomize(bool on)
{
    step_order_randomize_ = on;
    lm::Print::printf(lm::Print::INFO, on
        ? "Step operator order: RANDOMIZED per step (x, y, z drift, structures+field, reactions; fused z+reaction off)."
        : "Step operator order: fixed (x, y, z drift, structures, field, reactions).");
}

void DriftMpdRdmeSolver::setStructureSmearAfterMoves(bool on)
{
    struct_smear_after_moves_ = on;
    lm::Print::printf(lm::Print::INFO, on
        ? "Structure smear refresh: after the move group (bodies read where they are)."
        : "Structure smear refresh: at the head of the structure passes (default).");
}

void DriftMpdRdmeSolver::stepOrderPermutation(uint32_t timestep, int m, int* idx)
{
    for (int i = 0; i < m; ++i) idx[i] = i;
    if (!step_order_randomize_) return;
    for (int i = m - 1; i > 0; --i)
    {
        const uint64_t h = getTimestepSeed(timestep, kStepOrderSaltBase + (uint32_t)i);
        const int j = (int)(h % (uint64_t)(i + 1));
        const int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

}  // namespace rdme
}  // namespace lm
