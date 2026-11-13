#ifndef GP_RESIDUAL_MODEL_HPP
#define GP_RESIDUAL_MODEL_HPP

#include "StochasticSystemModel.hpp"
#include "models/BicycleModel.hpp"
#include <Eigen/Dense>
#include <filesystem>

namespace mpc {

class GPResidualModel : public StochasticSystemModel {
public:
    // Construct GP residual model from the JSON parameter file and the wheelbase.
    // Throws std::runtime_error naming the path if the file cannot be opened or parsed.
    GPResidualModel(const std::filesystem::path& params_path, double wheelbase);

    // SystemModel interface
    [[nodiscard]] Eigen::VectorXd dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    [[nodiscard]] Eigen::MatrixXd jacobian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    [[nodiscard]] Eigen::MatrixXd jacobian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;
    [[nodiscard]] int state_dim() const override;
    [[nodiscard]] int input_dim() const override;

    // StochasticSystemModel interface
    // Diagonal process noise variance, GP uncertainty placed on the heading rate
    [[nodiscard]] Eigen::VectorXd variance(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override;


private:
    BicycleModel bicycle_model_;     // delegates dynamics, Jacobians, and dimensions
    double sigma_f_sq_;              // kernel output scale
    double l_v_;                     // kernel length scale for velocity
    double l_delta_;                 // kernel length scale for steering angle
    Eigen::MatrixXd X_train_;        // training data
    Eigen::MatrixXd K_inv_;          // inverse of the kernel matrix
    Eigen::VectorXd alpha_;          // GP dual coefficients

    // GP posterior mean: the learened residual correction at operating point (v, delta)
    [[nodiscard]] double gp_mean(double v, double delta) const;
    // GP posterior variance: model uncertainty at operating point (v, delta)
    [[nodiscard]] double gp_variance(double v, double delta) const;

    // RBF kernel function: similarity between two input points [v, delta]
    [[nodiscard]] double kernel(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;

    // Partial derivates of the GP posterior mean (for the Jacobian contribution)
    [[nodiscard]] double gp_mean_dv(double v, double delta) const;
    [[nodiscard]] double gp_mean_ddelta(double v, double delta) const;
};

} // namespace mpc

#endif // GP_RESIDUAL_MODEL_HPP