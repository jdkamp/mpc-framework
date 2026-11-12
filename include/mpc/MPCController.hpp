#ifndef MPC_MPC_CONTROLLER_HPP
#define MPC_MPC_CONTROLLER_HPP

#include "SystemModel.hpp"
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <memory>

namespace mpc {

// Configuration parameters for the MPC controller
struct MPCConfig {
    int N = 0;      // Prediction horizon 
    double dt = 0.0;  // time step [s]
    
};

struct MPCWeights {
    Eigen::MatrixXd Q;  // Q weights
    Eigen::MatrixXd Qf; // terminal cost weight
    Eigen::MatrixXd R;  // R weights
    Eigen::MatrixXd S;  // S weights for delta_u (control input change)

    // Default to identity Q/Qf/R and zero S
    explicit MPCWeights(const SystemModel& model);

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
    explicit MPCLimits(const SystemModel& model);
};

// Outcome of one solve. The fallback criterion is unchanged: the solution is
// rejected if it is non-finite or diverged, and the previous input is held.
enum class SolveStatus {
    Solved,         // QP solution accepcted
    HeldPrevious    // solution rejected, previous input returned
};

struct [[nodiscard]] SolveResult {
    SolveStatus status;
    Eigen::VectorXd u;  // control input to apply
};

// MPC controller - solves a QP at each time step using a linearized system model
class MPCController {
public:
    // Construct MPC controller with a system model and mpc config
    MPCController(std::shared_ptr<const SystemModel> model, MPCConfig config, const MPCWeights& weights, const MPCLimits& limits);
    
    // Solve the MPC problem from state x0 tracking x_ref(n*N) over prediction horizon N
    // Returns the first optimal control input [delta, a]
    SolveResult solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& x_ref);
    
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
    // Model
    std::shared_ptr<const SystemModel> model_;
    // MPC inputs
    MPCConfig config_;
    MPCWeights weights_;
    MPCLimits limits_;
    // Solver
    OsqpEigen::Solver solver_;
    bool solver_initialized_ = false;
    Eigen::SparseMatrix<double> P_stored_, A_stored_;

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

} // namespace mpc

#endif // MPC_MPC_CONTROLLER_HPP