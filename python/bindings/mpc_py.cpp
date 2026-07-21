#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "models/GPResidualModel.hpp"
#include "models/BicycleModel.hpp"
#include "mpc/SafetyFilter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mpc_py, m) {
    m.doc() = "C++ MPC/safety-filter bindings";

    // Base classes: abstract, no constructors - registered so pybind knows
    // GPResidualModel is a StochasticSystemModel when matching signatures.
    py::class_<SystemModel>(m, "SystemModel");
    py::class_<StochasticSystemModel, SystemModel>(m, "StochasticSystemModel");

    py::class_<GPResidualModel, StochasticSystemModel>(m, "GPResidualModel")
        .def(py::init<const std::string&, double>(),
            py::arg("params_path"), py::arg("wheelbase"))
        .def("variance", &GPResidualModel::variance);

    py::class_<MPCConfig>(m, "MPCConfig")
        .def(py::init<>())
        .def_readwrite("N", &MPCConfig::N)
        .def_readwrite("dt", &MPCConfig::dt);

    py::class_<MPCWeights>(m, "MPCWeights")
        .def(py::init<const SystemModel&>())
        .def_readwrite("Q", &MPCWeights::Q)
        .def_readwrite("R", &MPCWeights::R)
        .def_readwrite("S", &MPCWeights::S)
        .def_readwrite("Qf", &MPCWeights::Qf);

    py::class_<MPCLimits>(m, "MPCLimits")
        .def(py::init<const SystemModel&>())
        .def_readwrite("u_min", &MPCLimits::u_min)
        .def_readwrite("u_max", &MPCLimits::u_max)
        .def_readwrite("x_min", &MPCLimits::x_min)
        .def_readwrite("x_max", &MPCLimits::x_max)
        .def_readwrite("delta_u_min", &MPCLimits::delta_u_min)
        .def_readwrite("delta_u_max", &MPCLimits::delta_u_max);

    py::class_<SafetyFilter>(m, "SafetyFilter")
        .def(py::init<const StochasticSystemModel&, MPCConfig,
                      const MPCWeights&, const MPCLimits&, double>(),
            py::arg("model"), py::arg("config"), py::arg("weights"),
            py::arg("limits"), py::arg("p"),
            py::keep_alive<1, 2>())          // the filter keeps the model alive
        .def("filter", &SafetyFilter::filter,
            py::arg("x0"), py::arg("u_rl"));

    py::class_<BicycleModel, SystemModel>(m, "BicycleModel")
        .def(py::init<double>(), py::arg("wheelbase"))
        .def("dynamics", &BicycleModel::dynamics,
            py::arg("x"), py::arg("u"));
}
