#include "models/ConstantVelocityAgent.hpp"

using Eigen::VectorXd;

ConstantVelocityAgent::ConstantVelocityAgent(double sigma0_sq, double q)
    : sigma0_sq_(sigma0_sq), q_(q) {}

void ConstantVelocityAgent::observe(double py, double vy) {
    py_ = py;
    vy_ = vy;
}

AgentPrediction ConstantVelocityAgent::predict(int N, double dt) const {
    AgentPrediction pred;
    pred.mean = VectorXd(N);
    pred.variance = VectorXd(N);

    for (int i = 0; i < N; i++) {
        double t = (i + 1) * dt;                // time ahead at step k = i+1
        pred.mean(i) = py_ + t * vy_;           // constant-velocity extrapolation
        pred.variance(i) = sigma0_sq_ + t * q_; // uncertainty grows linear
    }

    return pred;
}