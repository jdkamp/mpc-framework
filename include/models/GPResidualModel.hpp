#ifndef GP_RESIDUAL_MODEL_HPP
#define GP_RESIDUAL_MODEL_HPP

#include "StochasticSystemModel.hpp"
#include "models/BicycleModel.hpp"
#include <Eigen/Dense>
#include <string>

class GPResidualModel : public StochasticSystemModel {
public:
    // Construct GP residual model with the parameters path and wheelbase
    GPResidualModel(const std::string& params_path, double wheelbase);

    // SystemModel interface
    Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    int state_dim() const override;
    int input_dim() const override;

    // StochasticSystemModel interface
    // Diagonal process noise variance, GP uncertainty placed on the heading rate
    Eigen::VectorXd variance(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;


private:
    BicycleModel bicycle_model_;     // delegates dynamics, Jacobians, and dimensions
    double sigma_f_sq_;              // kernel output scale
    double l_v_;                     // kernel length scale for velocity
    double l_delta_;                 // kernel length scale for steering angle
    Eigen::MatrixXd X_train_;        // training data
    Eigen::MatrixXd K_inv_;          // inverse of the kernel matrix
    Eigen::VectorXd alpha_;          // GP dual coefficients

    // GP posterior mean: the learened residual correction at operating point (v, delta)
    double gp_mean(double v, double delta) const;
    // GP posterior variance: model uncertainty at operating point (v, delta)
    double gp_variance(double v, double delta) const;

    // RBF kernel function: similarity between two input points [v, delta]
    double kernel(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;

    // Partial derivates of the GP posterior mean (for the Jacobian contribution)
    double gp_mean_dv(double v, double delta) const;
    double gp_mean_ddelta(double v, double delta) const;
};

#endif // GP_RESIDUAL_MODEL_HPP