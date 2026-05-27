#ifndef MPC_BICYCLE_MODEL_HPP
#define MPC_BICYCLE_MODEL_HPP

#include "../SystemModel.hpp"
#include <Eigen/Dense>

class BicycleModel : public SystemModel {
public:
    BicycleModel(double wheelbase) : L_(wheelbase) {}
    Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    int state_dim() const override;
    int input_dim() const override;


private:
    double L_;
};

#endif // MPC_BICYCLE_MODEL_HPP
