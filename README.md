# MPC Framework

A generic Model Predictive Control (MPC) framework in C++17. The controller is decoupled from any specific system model via a `SystemModel` interface — plug in any linearizable model and the controller handles the rest.

## Features

- Generic `SystemModel` interface — implement once, reuse the controller unchanged
- Linearization-based MPC with Euler discretization at each step
- Quadratic cost with optional terminal weight
- Input constraints
- Predicted trajectory output from each solve
- Kinematic bicycle model as reference implementation
- Lane-keeping example
- Unit tests with Google Test

## Architecture

```
SystemModel (abstract interface)
    └── BicycleModel (kinematic, state: [px, py, ψ, v], input: [δ, a])

MPCController
    ├── takes SystemModel& and MPCConfig
    ├── linearizes model at each step → A, B matrices
    ├── builds lifted QP (Sx, Su, Q_bar, R_bar)
    └── solves via OSQP
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

## Run the lane-keeping example

```bash
mkdir -p output
./build/lane_keeping
```

Runs a 50-step simulation of a vehicle tracking a lane at 5 m/s with a lateral offset, and writes `output/trajectory.csv`.

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
tests/
    BicycleModelTest.cpp
    MPCControllerTest.cpp
scripts/
    visualize_trajectory.py
```
