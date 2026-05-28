#ifndef MPC_BICYCLE_MODEL_HPP
#define MPC_BICYCLE_MODEL_HPP

#include "../SystemModel.hpp"
#include <Eigen/Dense>

// Kinematic bicycle model — concrete implementation of SystemModel.
// State: [px, py, psi, v]  Input: [delta, a]
class BicycleModel : public SystemModel {
public:
    // Sets wheel base
    BicycleModel(double wheelbase) : L_(wheelbase) {}

    // SystemModel interface
    Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    int state_dim() const override;
    int input_dim() const override;


private:
    double L_;                          // wheel base
    static constexpr int STATE_DIM = 4; // state dimension xp, yp, psi, v
    static constexpr int INPUT_DIM = 2; // input dimension delta, a
};

#endif // MPC_BICYCLE_MODEL_HPP
