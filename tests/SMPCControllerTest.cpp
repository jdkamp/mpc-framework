#include <gtest/gtest.h>
#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"
#include "mpc/SMPCController.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector4d;

// Bicycle dynamics + a constant, known covariance on py, let test the tube
// logic with controlled uncertainty, independant of the GP
class ConsVarianceModel : public StochasticSystemModel {
    BicycleModel bicycle_model_{2.7};
    double var_;
public:
    ConsVarianceModel(double var) : var_(var) {}
    VectorXd dynamics(const VectorXd& x, const VectorXd& u) const override { 
        return bicycle_model_.dynamics(x, u);
    }
    MatrixXd jacobian_x(const VectorXd& x, const VectorXd& u) const override {
        return bicycle_model_.jacobian_x(x, u);
    }
    MatrixXd jacobian_u(const VectorXd& x, const VectorXd& u) const override {
        return bicycle_model_.jacobian_u(x, u);
    }
    int state_dim() const override {
        return bicycle_model_.state_dim();
    }
    int input_dim() const override{
        return bicycle_model_.input_dim();
    }
    VectorXd variance(const VectorXd& x, const VectorXd& u) const override {
        VectorXd v = VectorXd::Zero(4);
        v(1) = var_;    // uncertainty on py
        return v;
    }
};

// With zero variance, SMPC must reduce exactly to determenistic MPC (backoff = 0)
TEST(SMPCControllerTest, NoVarianceMatchesDeterminstic) {
    ConsVarianceModel model(0.0);
    BicycleModel det_model(2.7);

    MPCConfig config; 
    config.N = 30; 
    config.dt = 0.01;
    MPCWeights weights(det_model);
    weights.Q = Vector4d(0, 10, 50, 10).asDiagonal();
    weights.R = Vector2d(1, 10).asDiagonal();
    VectorXd x_trim = (VectorXd(4) << 0, 0, 0, 5).finished();
    weights.Qf = weights.compute_dare(det_model, x_trim, config.dt);

    MPCLimits limits(det_model);
    limits.x_max(1) = 1.0;

    VectorXd x0(4);    x0    << 0, 0, 0, 5;
    VectorXd x_ref(4); x_ref << 0, 2, 0, 5;

    MPCController det_mpc (det_model, config, weights, limits);
    SMPCController smpc(model, config, weights, limits, 0.95);

    det_mpc.solve(x0, x_ref.replicate(config.N, 1));
    smpc.solve(x0, x_ref.replicate(config.N, 1));

    EXPECT_TRUE(smpc.get_predicted_trajectory().isApprox(   // get predicted trajectories are equal
                det_mpc.get_predicted_trajectory(), 1e-6
    ));
}

// Non-zero variance must back the predicted trajectoy off from an active constraint
TEST(SMPCControllerTest, TighterThanNominal) {
    ConsVarianceModel model(0.001); // uncertainty on py
    BicycleModel det_model(2.7);

    MPCConfig config; 
    config.N = 30; 
    config.dt = 0.01;
    MPCWeights weights(det_model);
    weights.Q = Vector4d(0, 10, 50, 10).asDiagonal();
    weights.R = Vector2d(1, 10).asDiagonal();
    VectorXd x_trim = (VectorXd(4) << 0, 0, 0, 5).finished();
    weights.Qf = weights.compute_dare(det_model, x_trim, config.dt);

    MPCLimits limits(det_model);
    limits.x_max(1) = 1.0;

    VectorXd x0(4);    x0    << 0, 0, 0, 5;
    VectorXd x_ref(4); x_ref << 0, 2, 0, 5; // Push py up past the bound

    MPCController det_mpc (det_model, config, weights, limits);
    SMPCController smpc(model, config, weights, limits, 0.95);

    det_mpc.solve(x0, x_ref.replicate(config.N, 1));
    smpc.solve(x0, x_ref.replicate(config.N, 1));

    double detmpc_max = det_mpc.get_predicted_trajectory().col(1).maxCoeff();
    double smpc_max = smpc.get_predicted_trajectory().col(1).maxCoeff();

    EXPECT_LT(smpc_max, detmpc_max);    // SMPC backs off - stays further below the bound
}
