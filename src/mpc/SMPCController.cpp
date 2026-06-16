#include "mpc/SMPCController.hpp"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;

SMPCController::SMPCController(const StochasticSystemModel& model, MPCConfig config, 
                                const MPCWeights& weights, const MPCLimits& limits, double p) :
    MPCController(model, config, weights, limits),  // Base gets Systemmodel as refrence
    smodel_(model),                                 // keep the typed reference
    p_(p)
{}


VectorXd SMPCController::compute_backoff(const VectorXd& x0) const {
    int n = model_.state_dim();
    int N = config_.N;
    VectorXd u0 = VectorXd::Zero(model_.input_dim());

    // Linearize at x0 and discretize
    MatrixXd A_d = MatrixXd::Identity(n, n) + config_.dt * smodel_.jacobian_x(x0, u0);
    MatrixXd B_d = config_.dt * smodel_.jacobian_u(x0, u0);

    // Closed-loop dynamics using the LQR gain
    MatrixXd K = weights_.compute_lqr_gain(smodel_, x0, config_.dt);
    MatrixXd A_cl = A_d + B_d * K;

    // Process-noise covariance from the model's state-dependent variance
    MatrixXd Sigma_w = smodel_.variance(x0, u0).asDiagonal();

    // Cantelli factor for satisfaction probability p
    double c = std::sqrt(p_ / (1.0 - p_));

    // Propagate the tube and build the per-step backoff
    VectorXd backoff(n * N);
    MatrixXd Sigma = MatrixXd::Zero(n, n);          // Sigma_0 = 0 (current state known)
    for (int k = 0; k < N; k++) {
        Sigma = A_cl * Sigma * A_cl.transpose() + Sigma_w;  // Sigma_(k+1)
        VectorXd std_dev = Sigma.diagonal().cwiseSqrt();    // standard deviation per state
        backoff.segment(k * n, n) = c * std_dev;
    }

    return backoff;

}

VectorXd SMPCController::state_upper_bounds(const MatrixXd& Sx, const VectorXd& x0) const {
    return MPCController::state_upper_bounds(Sx, x0) - compute_backoff(x0);     // tighten down
}

VectorXd SMPCController::state_lower_bounds(const MatrixXd& Sx, const VectorXd& x0) const {
    return MPCController::state_lower_bounds(Sx, x0) + compute_backoff(x0);     // tighten up
}

