#ifndef MPC_SYSTEM_MODEL_HPP
#define MPC_SYSTEM_MODEL_HPP
#include <Eigen/Dense>

class SystemModel {
public:
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual int state_dim() const = 0;
    virtual int input_dim() const = 0;

    virtual ~SystemModel() = default;

};


#endif // MPC_SYSTEM_MODEL_HPP