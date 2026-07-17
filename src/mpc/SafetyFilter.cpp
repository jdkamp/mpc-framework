#include "mpc/SafetyFilter.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

SafetyFilter::SafetyFilter(const StochasticSystemModel& model, MPCConfig config,
                const MPCWeights& weights, const MPCLimits& limits, double p) :
    SMPCController(model, config, weights, limits, p),
    u_rl_(VectorXd::Zero(model.input_dim()))
{}

VectorXd SafetyFilter::filter(const VectorXd& x0, const VectorXd& u_rl) {
    u_rl_ = u_rl;
    return solve(x0, VectorXd::Zero(model_.state_dim() * config_.N));
}

void SafetyFilter::build_cost(const MatrixXd& Su,
                              const VectorXd& x_free,
                              const VectorXd& x_ref,
                              const MatrixXd& D,
                              const VectorXd& d_prev,
                              MatrixXd& P,
                              VectorXd& q) const {
    int m = model_.input_dim();
    int nU = m * config_.N;
    double eps = 1e-4;

    // min 0.5*||u0 - u_rl||^2  +  0.5*eps*||u_1..u_{N-1}||^2
    // in the same 0.5*U'PU + q'U convention as the base tracking cost:
    // expanding 0.5*||u0 - u_rl||^2 = 0.5*u0'*u0 - u_rl'*u0 + const
    P = eps * MatrixXd::Identity(nU, nU);              // tail: tie-breaker only
    P.topLeftCorner(m, m) = MatrixXd::Identity(m, m);  // u0 carries full weight
    q = VectorXd::Zero(nU);
    q.head(m) = -u_rl_;                                // the linear term; const dropped


}