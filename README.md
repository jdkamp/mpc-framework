# MPC Framework

A generic Model Predictive Control (MPC) framework in C++17. The controller is decoupled from any specific system model via a `SystemModel` interface — plug in any linearizable model and the controller handles the rest.

## Motivation

Classical MPC assumes a known, exact model. Real systems have unmodeled dynamics, parameter uncertainty, and disturbances that no hand-tuned noise matrix fully captures. This framework is built to be model-agnostic: the `SystemModel` interface is designed so that a physics-based model and a learned model (Gaussian Process, neural network) are interchangeable without changing the controller. The goal is to bridge standard SMPC formulations with data-driven uncertainty estimation — replacing assumed noise covariance with uncertainty learned from data.

## Features

- Generic `SystemModel` interface — implement once, reuse the controller unchanged
- Linearization-based MPC with Euler discretization at each step
- Quadratic cost with state weights (Q), input weights (R), rate weights (S), and DARE-computed terminal weight (Qf)
- Input constraints, state constraints, and control rate constraints
- Predicted trajectory output from each solve
- Kinematic bicycle model as reference implementation
- Lane-keeping and path-following examples
- Unit tests with Google Test

## Architecture

```
SystemModel (abstract interface)
    └── BicycleModel (kinematic, state: [px, py, ψ, v], input: [δ, a])

MPCController
    ├── takes SystemModel&, MPCConfig, MPCWeights, MPCLimits
    ├── linearizes model at each step → A, B matrices
    ├── builds lifted QP (Sx, Su, Q_bar, R_bar)
    └── solves via OSQP

MPCWeights
    ├── Q, R, S — state, input, and control rate cost matrices
    └── Qf — terminal cost, optionally computed via DARE

MPCLimits
    └── input, state, and control rate bounds
```

To use a different system, implement `SystemModel` and pass it to `MPCController` — no controller code changes needed.

## Dependencies

All dependencies are fetched automatically via CMake `FetchContent`:

| Library | Version | Purpose |
|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | 3.4.0 | Matrix algebra |
| [OSQP](https://osqp.org) | 1.0.0 | QP solver |
| [osqp-eigen](https://github.com/robotology/osqp-eigen) | 0.11.0 | C++ OSQP wrapper |
| [GoogleTest](https://github.com/google/googletest) | 1.14.0 | Unit testing |

Requires: CMake ≥ 3.14, a C++17 compiler.

## Build

```bash
mkdir build
cmake -S . -B build
cmake --build build
```

## Run the examples

**Lane-keeping:** tracks a constant lateral reference at 5 m/s.

```bash
mkdir -p output
./build/lane_keeping
```

**Path-following:** tracks a sinusoidal path at 5 m/s.

```bash
./build/path_following
```

Both write `output/trajectory.csv`.

Visualize with Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas matplotlib
python3 scripts/visualize_trajectory.py
```

## Run tests

```bash
./build/tests
```

## Implementing a custom model

Inherit from `SystemModel` and implement five methods:

```cpp
class MyModel : public SystemModel {
public:
    Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    int state_dim() const override { return ...; }
    int input_dim() const override { return ...; }
};
```

Then pass it to the controller:

```cpp
MyModel model(...);
MPCController mpc(model, config);
Eigen::VectorXd u = mpc.solve(x0, x_ref);
```

## Project structure

```
include/
    SystemModel.hpp          # abstract interface
    models/BicycleModel.hpp
    mpc/MPCController.hpp
src/
    models/BicycleModel.cpp
    mpc/MPCController.cpp
examples/
    lane_keeping/main.cpp
    path_following/main.cpp
tests/
    BicycleModelTest.cpp
    MPCControllerTest.cpp
scripts/
    visualize_trajectory.py
```
