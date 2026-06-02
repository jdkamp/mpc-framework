#ifndef MPC_MPC_CONTROLLER_HPP
#define MPC_MPC_CONTROLLER_HPP

#include "SystemModel.hpp"
#include <Eigen/Dense>

// Configuration parameters for the MPC controller
struct MPCConfig {
    int N;      // Prediction horizon 
    double dt;  // time step [s]
    
};

struct MPCWeights {
    Eigen::MatrixXd Q;  // Q weights
    Eigen::MatrixXd Qf; // terminal cost weight
    Eigen::MatrixXd R;  // R weights
    Eigen::MatrixXd S;  // S weights for delta_u (control input change)
    
    MPCWeights(const SystemModel& model) : // default to identity weights
        Q(Eigen::MatrixXd::Identity(model.state_dim(), model.state_dim())),
        Qf(Eigen::MatrixXd::Identity(model.state_dim(), model.state_dim())),
        R(Eigen::MatrixXd::Identity(model.input_dim(), model.input_dim())),
        S(Eigen::MatrixXd::Zero(model.input_dim(), model.input_dim()))
    {}
};

struct MPCLimits {
    Eigen::VectorXd u_min;       // minimum control inputex
    Eigen::VectorXd u_max;       // maximum control input
    Eigen::VectorXd x_min;       // minimum state
    Eigen::VectorXd x_max;       // maximum state
    Eigen::VectorXd delta_u_min; // minimum change in control input
    Eigen::VectorXd delta_u_max; // maximum change in control input

    MPCLimits(const SystemModel& model) : // default to no constraints
        u_min(Eigen::VectorXd::Constant(model.input_dim(), -1e10)),
        u_max(Eigen::VectorXd::Constant(model.input_dim(), 1e10)),
        x_min(Eigen::VectorXd::Constant(model.state_dim(), -1e10)),
        x_max(Eigen::VectorXd::Constant(model.state_dim(), 1e10)),
        delta_u_min(Eigen::VectorXd::Constant(model.input_dim(), -1e10)),
        delta_u_max(Eigen::VectorXd::Constant(model.input_dim(), 1e10))
    {}
};

// MPC controller - solves a QP at each time step using a linearized system model
class MPCController {
public:
    // Construct MPC controller with a system model and mpc config
    MPCController(const SystemModel& model, MPCConfig config, const MPCWeights& weights, const MPCLimits& limits);

    // Solve the MPC problem from state x0 tracking x_ref(n*N) over prediction horizon N
    // Returns the first optimal control input [delta, a]
    Eigen::VectorXd solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& x_ref);
    
    // Returns the predicted state trajectory from the last solve() call
    Eigen::MatrixXd get_predicted_trajectory() const;

    // Returns the time of the last solve() call in seconds
    double get_solve_time() const;
private:
    const SystemModel& model_;
    MPCConfig config_;
    MPCWeights weights_;
    MPCLimits limits_;
    Eigen::MatrixXd predicted_trajectory_;
    double solve_time_;
    Eigen::VectorXd previous_u_; // for delta_u constraints
};

#endif // MPC_MPC_CONTROLLER_HPP