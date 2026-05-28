#ifndef MPC_SYSTEM_MODEL_HPP
#define MPC_SYSTEM_MODEL_HPP
#include <Eigen/Dense>

// Abstract interface for system models used by MPCController.
// Every concrete model must implement dynamics and Jacobians.
class SystemModel {
public:
    // Returns state derivative ẋ = f(x, u) 
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    
    // Returns Jacobian of f w.r.t. state x (A matrix)
    virtual Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    
    // Returns Jacobian of f w.r.t. input u (B matrix) 
    virtual Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    
    // Returns state dimension
    virtual int state_dim() const = 0;
    
    // Returns input dimension
    virtual int input_dim() const = 0;

    virtual ~SystemModel() = default;

};


#endif // MPC_SYSTEM_MODEL_HPP