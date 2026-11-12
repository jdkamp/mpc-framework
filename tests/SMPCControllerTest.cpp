#include <gtest/gtest.h>
#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"
#include "mpc/SMPCController.hpp"
#include "models/ConstantVelocityAgent.hpp"
#include <memory>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector4d;
using namespace mpc;

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
    VectorXd variance(const VectorXd& /*x*/, const VectorXd& /*u*/) const override {
        VectorXd v = VectorXd::Zero(4);
        v(1) = var_;    // uncertainty on py
        return v;
    }
};

class SMPCControllerTest : public ::testing::Test {
protected:
std::shared_ptr<BicycleModel> det_model = std::make_shared<BicycleModel>(2.7);
    MPCConfig config;
    MPCWeights weights{*det_model};
    MPCLimits limits{*det_model};
    VectorXd x0{4}, x_ref{4};

    void SetUp() override {
        config.N = 30; config.dt = 0.01;
        weights.Q = Vector4d(0, 10, 50, 10).asDiagonal();
        weights.R = Vector2d(1, 10).asDiagonal();
        VectorXd x_trim = (VectorXd(4) << 0, 0, 0, 5).finished();
        weights.Qf = weights.compute_dare(*det_model, x_trim, config.dt);
        limits.x_max(1) = 1.0;
        x0 << 0, 0, 0, 5;
        x_ref << 0, 2, 0, 5;
    }
};

// With zero variance, SMPC must reduce exactly to deterministic MPC (backoff = 0)
TEST_F(SMPCControllerTest, NoVarianceMatchesDeterministic) {
    auto model = std::make_shared<ConsVarianceModel>(0.0);

    MPCController det_mpc(det_model, config, weights, limits);
    SMPCController smpc(model, config, weights, limits, 0.95);

    ASSERT_EQ(det_mpc.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);
    ASSERT_EQ(smpc.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);

    EXPECT_TRUE(smpc.get_predicted_trajectory().isApprox(
                det_mpc.get_predicted_trajectory(), 1e-6
    ));
}

// Non-zero variance must back the predicted trajectory off from an active constraint
TEST_F(SMPCControllerTest, TighterThanNominal) {
    auto model = std::make_shared<ConsVarianceModel>(0.001);    // uncertainty on py

    MPCController det_mpc(det_model, config, weights, limits);
    SMPCController smpc(model, config, weights, limits, 0.95);

    ASSERT_EQ(det_mpc.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);
    ASSERT_EQ(smpc.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);

    double detmpc_max = det_mpc.get_predicted_trajectory().col(1).maxCoeff();
    double smpc_max = smpc.get_predicted_trajectory().col(1).maxCoeff();

    EXPECT_LT(smpc_max, detmpc_max);    // SMPC backs off - stays further below the bound
}

// An agent far from the ego must not change the solution
TEST_F(SMPCControllerTest, FarAgentIsNoOp) {
    auto model = std::make_shared<ConsVarianceModel>(0.001);

    SMPCController smpc_no_agent(model, config, weights, limits, 0.95);
    SMPCController smpc_agent(model, config, weights, limits, 0.95);

    ConstantVelocityAgent agent(0.01, 0.1);     // sigma0^2 = 0.01, growth q = 0.1
    agent.observe(100.0, 0.0);                  // at py = 100, far from the ego
    smpc_agent.set_agent(&agent, 0.5);          // r_safe = 0.5

    ASSERT_EQ(smpc_no_agent.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);
    ASSERT_EQ(smpc_agent.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);

    EXPECT_TRUE(smpc_agent.get_predicted_trajectory().isApprox(
                smpc_no_agent.get_predicted_trajectory(), 1e-9));
}

// An agent above the ego must tighten the py bound below the static limit
TEST_F(SMPCControllerTest, AgentTightensBound) {
    auto model = std::make_shared<ConsVarianceModel>(0.001);

    SMPCController smpc_no_agent(model, config, weights, limits, 0.95);
    SMPCController smpc_agent(model, config, weights, limits, 0.95);

    ConstantVelocityAgent agent(0.01, 0.1);     // sigma0^2 = 0.01, growth q = 0.1
    agent.observe(1.0, 0.0);                    // at py = 1.0, not moving
    smpc_agent.set_agent(&agent, 0.5);          // r_safe = 0.5 -> bound near 0.5, under x_max = 1.0

    ASSERT_EQ(smpc_no_agent.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);
    ASSERT_EQ(smpc_agent.solve(x0, x_ref.replicate(config.N, 1)).status, SolveStatus::Solved);

    double no_agent_max = smpc_no_agent.get_predicted_trajectory().col(1).maxCoeff();
    double agent_max = smpc_agent.get_predicted_trajectory().col(1).maxCoeff();

    EXPECT_LT(agent_max, no_agent_max);     // the agent pushes the ego further down
}
