#ifndef STOCHASTIC_SYSTEM_MODEL_HPP
#define STOCHASTIC_SYSTEM_MODEL_HPP

#include "SystemModel.hpp"
#include <Eigen/Dense>

class StochasticSystemModel : public SystemModel {
public:
    // Returns the variance of the system model
    virtual Eigen::VectorXd variance(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;

};

#endif // STOCHASTIC_SYSTEM_MODEL_HPP