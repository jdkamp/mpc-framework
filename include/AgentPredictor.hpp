#ifndef AGENT_PREDICTOR_HPP
#define AGENT_PREDICTOR_HPP
#include <Eigen/Dense>

namespace mpc {

// Per-step predection of another agent's lateral position over the horizn.
struct  AgentPrediction{
    Eigen::VectorXd mean;       // py_agent at steps k = 1..N
    Eigen::VectorXd variance;   // var_agent at the steps
};

class AgentPredictor {
public:
    // Feed the agent's latest observed state
    virtual void observe(double py, double vy) = 0;

    // Predict N steps ahead with the size dt.
    [[nodiscard]] virtual AgentPrediction predict(int N, double dt) const = 0;

    virtual ~AgentPredictor() = default;
};


} // namespace mpc

#endif // AGENT_PREDICTOR_HPP