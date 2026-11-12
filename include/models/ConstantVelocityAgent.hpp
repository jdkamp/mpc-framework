#ifndef CONSTANT_VELOCITY_AGENT_HPP
#define CONSTANT_VELOCITY_AGENT_HPP

#include "AgentPredictor.hpp"
#include <Eigen/Dense>

namespace mpc {

class ConstantVelocityAgent : public AgentPredictor {
public:
    ConstantVelocityAgent(double sigma0_sq, double q);      // initial variance, growth rate
    void observe(double py, double vy) override;
    AgentPrediction predict(int N, double dt) const override;
private:
    double py_{0.0}, vy_{0.0};
    double sigma0_sq_, q_;
};

} // namespace mpc

#endif // CONSTANT_VELOCITY_AGENT_HPP