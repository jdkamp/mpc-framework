#ifndef MPC_SAFETY_FILTER_HPP
#define MPC_SAFETY_FILTER_HPP

#include "mpc/SMPCController.hpp"

// Safety Filter: same tightening as the SMPC but the objective is minimal
// intervention. Stay as close as possible to the agent's proposied actions
// instead of tracling reference.
class SafetyFilter : public SMPCController {
public:
    SafetyFilter(const StochasticSystemModel& model, MPCConfig config,
                 const MPCWeights& weights, const MPCLimits& limits, double p);

    // Filter a proposed actions: return the closest safe action to u_rl
    Eigen::VectorXd filter(const Eigen::VectorXd& x0, const Eigen::VectorXd& u_rl);

protected:
    void build_cost(const Eigen::MatrixXd& Su,
                    const Eigen::VectorXd& x_free,
                    const Eigen::VectorXd& x_ref,
                    const Eigen::MatrixXd& D,
                    const Eigen::VectorXd& d_prev,
                    Eigen::MatrixXd& P,
                    Eigen::VectorXd& q) const override;

private:
    Eigen::VectorXd u_rl_;  // stashed proposal, used in build cost
};

#endif // MPC_SAFETY_FILTER_HPP