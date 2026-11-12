#ifndef MPC_SMPC_CONTROLLER_HPP
#define MPC_SMPC_CONTROLLER_HPP

#include "mpc/MPCController.hpp"
#include "StochasticSystemModel.hpp"
#include "AgentPredictor.hpp"
#include <Eigen/Dense>
#include <memory>

namespace mpc {

// Stochastic MPC: propagates a closed-loop covariance tube along the prediction horizn
// and tightens the state constraints via the Cantelli inequality, using the model's state-
// dependant variance (e.g. GP-learned residual uncertainty)
class SMPCController : public MPCController {
public:
    // p = required chance-constraint satisfaction probability (e.g. 0.95)
    SMPCController(std::shared_ptr<const StochasticSystemModel> model, MPCConfig config, 
                   const MPCWeights& weights, const MPCLimits& limits, double p);
    // Non-owning: the caller must keep the predictor alive while this controller uses it.
    // Pass nullptr to remove the agent constraint.
    void set_agent(const AgentPredictor* agent, double r_safe);

protected:
    [[nodiscard]] Eigen::VectorXd state_upper_bounds(const Eigen::VectorXd& x_free, const Eigen::VectorXd& x0) const override;
    [[nodiscard]] Eigen::VectorXd state_lower_bounds(const Eigen::VectorXd& x_free, const Eigen::VectorXd& x0) const override;

private:
    std::shared_ptr<const StochasticSystemModel> smodel_;   // typed reference
    double p_;                              // chance-constraint satisfaction probability
    const AgentPredictor* agent_ = nullptr;
    double r_safe_ = 0.0;

    // Per-step constraint backoff (length n*N) from the propagated covariance tube
    [[nodiscard]] Eigen::VectorXd compute_backoff(const Eigen::VectorXd& x0) const;
    // Stacked diag(Sigma_k), k = 1..N
    [[nodiscard]] Eigen::VectorXd tube_variances(const Eigen::VectorXd& x0) const;
};

} // namespace mpc

#endif // MPC_SMPC_CONTROLLER_HPP