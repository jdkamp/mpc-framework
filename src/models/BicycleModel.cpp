#include "models/BicycleModel.hpp"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;

VectorXd BicycleModel::dynamics(const VectorXd& x, const VectorXd& u) const {
    // system states
    double px    = x(0);    // x-position
    double py    = x(1);    // y-position
    double psi   = x(2);    // yaw angle
    double vel   = x(3);    // vehicle velocity

    // inputs
    double delta = u(0);    // steering angle
    double a     = u(1);    // longitudinal acceleration

    // vehicle dynamics
    VectorXd xdot(4);
    xdot(0) = vel * std::cos(psi);
    xdot(1) = vel * std::sin(psi);
    xdot(2) = vel * std::tan(delta) / L_;
    xdot(3) = a;

    return xdot;
}

MatrixXd BicycleModel::jacobian_x(const VectorXd& x, const VectorXd& u) const {
    // system states
    double psi   = x(2);    // yaw angle
    double vel   = x(3);    // vehicle velocity

    // inputs
    double delta = u(0);    // steering angle
 
    // Jacobian x
    MatrixXd A = MatrixXd::Zero(4,4);
    A(0,2) = -vel * std::sin(psi);
    A(0,3) = std::cos(psi);
    A(1,2) = vel * std::cos(psi);
    A(1,3) = std::sin(psi);
    A(2,3) = std::tan(delta) / L_;

    return A;
}

MatrixXd BicycleModel::jacobian_u(const VectorXd& x, const VectorXd& u) const {
    // system states
    double vel   = x(3);    // vehicle velocity

    // inputs
    double delta = u(0);    // steering angle
 
    // Jacobian u
    MatrixXd B = MatrixXd::Zero(4,2);
    B(2,0) = vel / (L_ * std::pow(std::cos(delta),2));
    B(3,1) = 1;
    
    return B;
}

int BicycleModel::state_dim() const {
    return STATE_DIM;
}

int BicycleModel::input_dim() const {
    return INPUT_DIM;
}