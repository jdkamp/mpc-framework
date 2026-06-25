# MPC Framework

A generic Model Predictive Control (MPC) framework in C++17 that bridges classical Stochastic MPC with data-driven uncertainty.

## Motivation

Classical MPC assumes a known, exact model. Real systems have unmodeled dynamics, parameter uncertainty, and disturbances that no hand-tuned noise matrix fully captures. The conventional approach to add robustness — detuning the controller — buys safety with *uniform* conservatism. Performance is given up everywhere, even where the model is perfectly accurate.

This project takes the opposite approach. A Gaussian Process learns the model's residual and its own uncertainty, and the controller tightens its constraints in proportion to that uncertainty. Conservatism **only where the learned model is not trustworthy**, full performance everywhere else. The `SystemModel` interface keeps this model-agnostic: a physical model and a learned model (GP, neural network) are interchangeable without touching the controller.


## Results

Three controllers drive the same vehicle under disturbances, with **identical weights and limits** — so any difference comes from constraint handling alone:

| Controller | Model knowledge | Handles uncertainty | Violations |
|---|---|---|---|
| Deterministic MPC | nominal | no | 18.1% |
| Gaussian Process MPC (GP-MPC) | + learned mean | no | 24.7% |
| **Stochastic MPC (SMPC)** | + learned variance | yes | **0.85%** |

Over 10,000 disturbed steps, SMPC cuts constraint violations ~20×. Note that GP-MPC is *worse* than the deterministic controller, because it trusts a learned model's mean where it's uncertain. 

## Features

**Stochastic MPC with learned uncertainty**
- `StochasticSystemModel` — interface for models that provide state-dependent variance
- `GPResidualModel` — a Gaussian Process learning the model's residual *and* its own uncertainty from data. The GP mean (and its analytic derivative) enter the controller's linearization
- `SMPCController` — propagates a closed-loop covariance tube and tightens state constraints via the Cantelli inequality, turning GP variance into chance constraints
- Comparison demo (Deterministic MPC vs GP-MPC vs SMPC) and a Monte-Carlo violation-rate evaluation under disturbance

**Core MPC**
- Generic `SystemModel` interface — implement once, reuse the controller unchanged
- Linearization-based MPC with Euler discretization at each step
- Quadratic cost: state (Q), input (R), and input-rate (S) weights, and a DARE-computed terminal weight (Qf)
- Input, state, and control-rate constraints
- Kinematic bicycle model as the reference implementation
- Lane-keeping and path-following examples
- Unit tests with Google Test

## Architecture

```
SystemModel (abstract: dynamics, Jacobians, dims)
    ├── BicycleModel (kinematic, state: [px, py, ψ, v], input: [δ, a])
    └── StochasticSystemModel (adds variance(x,u) - state-dependent uncertainty)
        └── GPResidualModel (bicycle + GP-learned residual & variance)

MPCController
    ├── takes SystemModel&, MPCConfig, MPCWeights, MPCLimits
    ├── linearizes model at each step → A, B matrices
    ├── builds lifted QP (Sx, Su, Q_bar, R_bar)
    ├── solves via OSQP
    └── SMPCController
        ├── takes a StochasticSystemModel& and probability p
        ├── propagates a closed-loop covariance tube along the horizon
        └── tightens the state bound by Cantelli backoff (chance constraints)

MPCWeights - Q, R, S cost matrices, Qf terminal cost (DARE)
MPCLimits  - input, state, control-rate bounds
```

To use a different system, implement `SystemModel` (or `StochasticSystemModel` for an uncertainty-aware one) and pass it to `MPCController` / `SMPCController` — no controller code changes needed.

## Dependencies

All dependencies are fetched automatically via CMake `FetchContent`:

| Library | Version | Purpose |
|---|---|---|
| [Eigen](https://eigen.tuxfamily.org) | 3.4.0 | Matrix algebra |
| [OSQP](https://osqp.org) | 1.0.0 | QP solver |
| [osqp-eigen](https://github.com/robotology/osqp-eigen) | 0.11.0 | C++ OSQP wrapper |
| [GoogleTest](https://github.com/google/googletest) | 1.14.0 | Unit testing |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.x | Parse the exported GP parameters |


Requires: CMake ≥ 3.14, a C++17 compiler.

## Build

```bash
cmake -S . -B build
cmake --build build
mkdir -p data output
```

## Python environment (for GP training and plots)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib
```

## Reproduce the results

1. **Train the GP** — generates `data/gp_params.json` (required by the GP controllers) and the mean/variance surface plot `output/gp_surfaces.png`:
```bash
python3 scripts/generate_gp_training_data.py
python3 scripts/gp_training_data.py
```

2. **Comparison demo** — Deterministic vs. GP-MPC vs. SMPC under disturbance:
```bash
./build/comparison
python3 scripts/visualize_comparison.py    # -> output/comparison.png
```

3. **Monte-Carlo evaluation** — violation rates and the trajectory ensemble:
```bash
./build/montecarlo
python3 scripts/visualize_mc.py            # -> output/mc_violations.png
python3 scripts/visualize_ensemble.py      # -> output/ensemble.png
```

## Other examples
Basic single-controller tracking (no GP):
```bash
./build/lane_keeping
python3 scripts/visualize_trajectory.py
```
```bash
./build/path_following
python3 scripts/visualize_trajectory.py
```

## Tests
```bash
./build/tests
```


## Project structure

```
include/
    SystemModel.hpp             # abstract interface
    StochasticSystemModel.hpp   # + variance(x,u)
    models/
        BicycleModel.hpp
        GPResidualModel.hpp     # GP-learned residual + uncertainty
    mpc/
        MPCController.hpp
        SMPCController.hpp      # covariance tube + Cantelli tightening
src/
    models/
        BicycleModel.cpp
        GPResidualModel.cpp
    mpc/
        MPCController.cpp
        SMPCController.cpp
examples/
    lane_keeping/main.cpp
    path_following/main.cpp
    comparison/main.cpp         # Deterministic vs GP-MPC vs SMPC
    montecarlo/main.cpp         # violation-rate evaluation
tests/
    BicycleModelTest.cpp
    MPCControllerTest.cpp
    GPResidualModelTest.cpp
    SMPCControllerTest.cpp
scripts/
    generate_gp_training_data.py
    gp_training_data.py         # trains GP, exports data/gp_params.json
    visualize_trajectory.py
    visualize_comparison.py
    visualize_mc.py
    visualize_ensemble.py
```

## Limitations and future work

**Current Limitations**
- **Hard state constraints:** The `py` bound is enforced as a hard constraint, so a large disturbance can make the QP infeasible. Softening it with a penalized slack variable would keep the problem always feasible — the standard robust-MPC solution.
- **No solver-fault handling:** A failed or infeasible OSQP solve is currently used as-is. A robust controller should detect this and use a fall back strategy.
- **Linearization at zero input:** The model is re-linearized around `u0 = 0` each step rather than the actual operating point, which reduces prediction accuracy.

**Future work**
- **Robustness:** soft (chance) constraints + solver guarding, to remove the disturbance-induced infeasibility.
- **Neural-network models:** the `StochasticSystemModel` interface is model-agnostic, so a neural network that reports its own uncertainty (deep ensemble, MC-dropout, or a heteroscedastic head) could replace the GP **without touching the controller**.
- **Other system models:** extend beyond the kinematic bicycle model.