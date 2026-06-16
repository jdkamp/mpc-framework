#ifndef MPC_SMPC_CONTROLLER_HPP
#define MPC_SMPC_CONTROLLER_HPP

#include "mpc/MPCController.hpp"
#include "StochasticSystemModel.hpp"
#include <Eigen/Dense>

// Stochastic MPC: propagates a closed-loop covariance tube along the prediction horizn
// and tightens the state constraints via the Cantelli inequality, using the model's state-
// dependant variance (e.g. GP-learned residual uncertainty)
class SMPCController : public MPCController {
public:
    // p = required chance-constraint satisfaction probability (e.g. 0.95)
    SMPCController(const StochasticSystemModel& model, MPCConfig config, 
                   const MPCWeights& weights, const MPCLimits& limits, double p);

protected:
    Eigen::VectorXd state_upper_bounds(const Eigen::MatrixXd& Sx, const Eigen::VectorXd& x0) const override;
    Eigen::VectorXd state_lower_bounds(const Eigen::MatrixXd& Sx, const Eigen::VectorXd& x0) const override;

private:
    const StochasticSystemModel& smodel_;   // typed reference
    double p_;                              // chance-constraint satisfaction probability

    // Per-step constraint backoff (length n*N) from the propagated covariance tube
    Eigen::VectorXd compute_backoff(const Eigen::VectorXd& x0) const;
};

#endif // MPC_SMPC_CONTROLLER_HPP