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

    // Default to identity Q/Qf/R and zero S
    MPCWeights(const SystemModel& model);

    // Compute optimal terminal cost Qf via DARE using the current Q and R
    Eigen::MatrixXd compute_dare(
        const SystemModel& model,
        const Eigen::VectorXd& x_trim,
        const Eigen::VectorXd& u_trim,
        double dt,
        int max_iter = 1000,
        double tol = 1e-8) const;

    // Overload: assume zero input trim
    Eigen::MatrixXd compute_dare(
        const SystemModel& model,
        const Eigen::VectorXd& x_trim,
        double dt,
        int max_iter = 1000,
        double tol = 1e-8) const;

    // Compute the LQR feedback gain (such that u = K x) from the DARE solution
    // K = -(R + B_d^T P B_d)^-1 B_d^T P A_d
    Eigen::MatrixXd compute_lqr_gain(
        const SystemModel& model,
        const Eigen::VectorXd& x_trim,
        const Eigen::VectorXd& u_trim,
        double dt,
        int max_iter = 1000,
        double tol = 1e-8) const;

    // Overload. assume zero inputs trim
    Eigen::MatrixXd compute_lqr_gain(
        const SystemModel& model,
        const Eigen::VectorXd& x_trim,
        double dt,
        int max_xiter = 1000,
        double tol = 1e-8) const;

};

struct MPCLimits {
    Eigen::VectorXd u_min;       // minimum control input
    Eigen::VectorXd u_max;       // maximum control input
    Eigen::VectorXd x_min;       // minimum state
    Eigen::VectorXd x_max;       // maximum state
    Eigen::VectorXd delta_u_min; // minimum change in control input
    Eigen::VectorXd delta_u_max; // maximum change in control input

    // Default to no constraints (+/- 1e10)
    MPCLimits(const SystemModel& model);
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
    
    virtual ~MPCController() = default;

private:
    Eigen::MatrixXd predicted_trajectory_;
    double solve_time_;
    Eigen::VectorXd previous_u_; // for delta_u constraints

protected:
    const SystemModel& model_;
    MPCConfig config_;
    MPCWeights weights_;
    MPCLimits limits_;

    // Per-step state bounds for the lifted QP, shifted to the input variables.
    // x_free is the predicted free response (Sx*x0 + lifted affine offset, i.e. U = 0).
    // MPCController returns the untightened limits; the SMPCController overrides these
    // to apply chance-constraint tightening based on the propagated covariance tube.
    virtual Eigen::VectorXd state_upper_bounds(const Eigen::VectorXd& x_free, const Eigen::VectorXd& x0) const;
    virtual Eigen::VectorXd state_lower_bounds(const Eigen::VectorXd& x_free, const Eigen::VectorXd& x0) const;

    // Builds the QP cost over the inputs, before the slack augmentation. Base: reference
    // tracking. Subclasses override to change the objective (e.g. Safety Filter: stay close
    // to proposed action).
    virtual void build_cost(const Eigen::MatrixXd& Su,
                            const Eigen::VectorXd& x_free,
                            const Eigen::VectorXd& x_ref,
                            const Eigen::MatrixXd& D,
                            const Eigen::VectorXd& d_prev,
                            Eigen::MatrixXd& P,
                            Eigen::VectorXd& q) const;
};

#endif // MPC_MPC_CONTROLLER_HPP