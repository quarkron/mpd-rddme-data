// Sinusoidal one-body potential for ReaDDy: Phi(x) = gamma * sin(2 pi x / lambda + phase),
// acting on the x coordinate only. The field arm of the LM-RDDME comparison, matching the
// lattice's setPotentialBasis sinusoid exactly.
#include <pybind11/pybind11.h>
#include <cmath>
#include <readdy/model/potentials/PotentialOrder1.h>

namespace py = pybind11;

class Sinusoid : public readdy::model::potentials::PotentialOrder1 {
public:
    Sinusoid(readdy::ParticleTypeId ptype, readdy::scalar gamma, readdy::scalar wavelength,
             readdy::scalar phase)
        : PotentialOrder1(ptype), gamma(gamma), k(2.0 * M_PI / wavelength), phase(phase) {}

    std::string describe() const override { return "Sinusoidal one-body potential along x"; }
    std::string type() const override { return "Sinusoid"; }

    readdy::scalar calculateEnergy(const readdy::Vec3 &pos) const override {
        return gamma * std::sin(k * pos[0] + phase);
    }

    void calculateForce(readdy::Vec3 &force, const readdy::Vec3 &pos) const override {
        force += {-gamma * k * std::cos(k * pos[0] + phase), 0.0, 0.0};
    }

private:
    readdy::scalar gamma, k, phase;
};

PYBIND11_MODULE(readdy_sinpot, m) {
    py::module::import("readdy._internal.readdybinding.api");
    py::class_<Sinusoid, readdy::model::potentials::PotentialOrder1>(m, "Sinusoid")
        .def(py::init<readdy::ParticleTypeId,
                      readdy::scalar, readdy::scalar, readdy::scalar>());
}
