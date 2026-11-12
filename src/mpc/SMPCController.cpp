#include "mpc/SMPCController.hpp"
#include <cmath>
#include <algorithm>
#include <utility>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace mpc {

SMPCController::SMPCController(std::shared_ptr<const StochasticSystemModel> model, MPCConfig config, 
                                const MPCWeights& weights, const MPCLimits& limits, double p) :
    MPCController(model, config, weights, limits),  // Base gets Systemmodel as refrence
    smodel_(std::move(model)),                                 // keep the typed reference
    p_(p)
{}

VectorXd SMPCController::tube_variances(const VectorXd& x0) const {
    int n = model_->state_dim();
    int N = config_.N;
    VectorXd u0 = VectorXd::Zero(model_->input_dim());

    // Linearize at x0 and discretize
    MatrixXd A_d = MatrixXd::Identity(n, n) + config_.dt * smodel_->jacobian_x(x0, u0);
    MatrixXd B_d = config_.dt * smodel_->jacobian_u(x0, u0);

    // Closed-loop dynamics using the LQR gain
    MatrixXd K = weights_.compute_lqr_gain(*smodel_, x0, config_.dt);
    MatrixXd A_cl = A_d + B_d * K;

    // Process-noise covariance from the model's state-dependent variance
    MatrixXd Sigma_w = smodel_->variance(x0, u0).asDiagonal();

    // Propagate the tube and build the per-step backoff
    VectorXd var(n * N);
    MatrixXd Sigma = MatrixXd::Zero(n, n);          // Sigma_0 = 0 (current state known)
    for (int k = 0; k < N; k++) {
        Sigma = A_cl * Sigma * A_cl.transpose() + Sigma_w;  // Sigma_(k+1)
        var.segment(k * n, n) = Sigma.diagonal();           // variance not sqrt
    }

    return var;
}

VectorXd SMPCController::compute_backoff(const VectorXd& x0) const {
    double c = std::sqrt(p_ / (1.0 - p_));          // Cantelli factor for satisfaction probability p
    return c * tube_variances(x0).cwiseSqrt();      // backoff = c * sigma

}

VectorXd SMPCController::state_upper_bounds(const VectorXd& x_free, const VectorXd& x0) const {
    VectorXd ub = MPCController::state_upper_bounds(x_free, x0) - compute_backoff(x0);  // tighten down

    if (agent_) {
        int n = model_->state_dim();
        int N = config_.N;
        double c = std::sqrt(p_ / (1.0 - p_));
        AgentPrediction pred = agent_->predict(N, config_.dt);
        VectorXd tube_var = tube_variances(x0);

        for (int k = 0; k < N; k++) {
            double combined = c * std::sqrt(tube_var(k*n + 1) + pred.variance(k));  // backoff on combined std dev: tube + agent variances
            double agent_ub = pred.mean(k) - r_safe_ - combined - x_free(k*n + 1);  // agent bound on py, shifted into QP's deviation space (-x_free)
            ub(k*n + 1) = std::min(ub(k*n + 1), agent_ub);                          // tighter bound wins
        }

    }

    return ub;
}

VectorXd SMPCController::state_lower_bounds(const VectorXd& x_free, const VectorXd& x0) const {
    // Stage A assumption: the agent approaches from above, so it only tightens the
    // upper py bound (ego passes below) - the side choice that makes avoidance convex.
    // An agent from below would mirror here; choosing the side online is future work.
    return MPCController::state_lower_bounds(x_free, x0) + compute_backoff(x0);     // tighten up
}


void SMPCController::set_agent(const AgentPredictor* agent, double r_safe) {
    agent_ = agent;
    r_safe_ = r_safe;
}

} // namespace mpc
