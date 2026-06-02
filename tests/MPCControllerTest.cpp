#include <gtest/gtest.h>
#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Vector2d;

class MPCControllerTest : public ::testing::Test {
protected:
    BicycleModel model{2.7};
    
    MPCConfig config;
    MPCWeights weights{model};
    MPCLimits limits{model};

    void SetUp() override {
        config.N = 30; // prediction horizon
        config.dt = 0.1; // time step
        weights.Q = MatrixXd::Zero(4, 4);
        weights.Q(0,0) = 10.0;  // px — longitudinal tracking
        weights.Q(1,1) = 10.0;  // py — lane tracking
        weights.Q(2,2) = 50.0;  // psi — heading
        weights.Q(3,3) = 10.0;  // v — velocity
        weights.Qf = weights.Q;
        weights.R = MatrixXd::Zero(2, 2);
        weights.R(0,0) = 1.0;   // delta
        weights.R(1,1) = 10.0; // a acceleration
        weights.S = MatrixXd::Zero(2, 2);  // no rate cost by default
        limits.u_min = (VectorXd(2) << -1e10, -1e10).finished();  // no constraints
        limits.u_max = (VectorXd(2) << 1e10, 1e10).finished();  // no constraints
        limits.x_min = (VectorXd(4) << -1e10, -1e10, -1e10, -1e10).finished();  // no constraints
        limits.x_max = (VectorXd(4) << 1e10, 1e10, 1e10, 1e10).finished();  // no constraints
        limits.delta_u_min = (VectorXd(2) << -1e10, -1e10).finished();  // no constraints
        limits.delta_u_max = (VectorXd(2) << 1e10, 1e10).finished();  // no constraints
    }
};

TEST_F(MPCControllerTest, NoControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref = x0;  // reference state is the same as initial state

    MPCController mpc(model, config, weights, limits);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_NEAR(u(0), 0, 1e-6);
    EXPECT_NEAR(u(1), 0, 1e-6);
}

TEST_F(MPCControllerTest, PxControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);

    // Test positive px deviation
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 100, 0, 0, 5 ;  // px, py, psi, vel

    MPCController mpc(model, config, weights, limits);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(1), 0);  // u(1) > 0

    // Test negative px deviation
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << -100, 0, 0, 0;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(1), 0);  // u(1) < 0
}

TEST_F(MPCControllerTest, PyControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);

    // Test positive py deviation
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 1, 0, 5;  // px, py, psi, vel

    MPCController mpc(model, config, weights, limits);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    // Test negative py deviation
    EXPECT_GT(u(0), 0);  // u(0) > 0

    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, -1, 0, 5;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(0), 0);  // u(0) < 0
}

TEST_F(MPCControllerTest, PsiControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);

    // Test positive psi deviation
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 0, 1, 5;  // px, py, psi, vel

    MPCController mpc(model, config, weights, limits);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(0), 0);  // u(0) > 0

    // Test negative psi deviation
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 0, -1, 5;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(0), 0);  // u(0) < 0
}

TEST_F(MPCControllerTest, VelControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);

    // Test positive velocity deviation
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << 0, 0, 0, 1;  // px, py, psi, vel

    MPCController mpc(model, config, weights, limits);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(1), 0);  // u(1) > 0

    // Test negative velocity deviation
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << 0, 0, 0, -1;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(1), 0);  // u(1) < 0
}

TEST_F(MPCControllerTest, InputConstraints) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 100, 100, 100, 100;  // px, py, psi, vel

    // Without input constraints, verify unconstrained input is large enough to matter
    MPCController mpc_free(model, config, weights, limits);
    VectorXd u = mpc_free.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GE(u(0), 0.1);  // u(0) >= u_min(0)
    EXPECT_GE(u(1), 0.5);  // u(1) >= u_min(1)

    // With input constraints
    limits.u_min = (VectorXd(2) << -0.1, -0.5).finished();  // max steering angle, max acceleration
    limits.u_max = (VectorXd(2) << 0.1, 0.5).finished();  // min steering angle, min acceleration

    MPCController mpc_constrained(model, config, weights, limits);
    u = mpc_constrained.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LE(u(0), limits.u_max(0) + 1e-3);  // u(0) <= u_max(0) + small tolerance
    EXPECT_LE(u(1), limits.u_max(1) + 1e-3);  // u(1) <= u_max(1) + small tolerance
}

TEST_F(MPCControllerTest, RateConstraints) {
    VectorXd x0(4); x0 << 0, 0, 0, 5;
    VectorXd x_ref(4); x_ref << 0, 1, 1, 5;

    // Without rate constraints, verify unconstrained input is large enough to matter
    MPCController mpc_free(model, config, weights, limits);
    VectorXd u_free = mpc_free.solve(x0, x_ref.replicate(config.N, 1));
    EXPECT_GT(std::abs(u_free(0)), 0.05);

    // With tight rate constraints
    limits.delta_u_min = (VectorXd(2) << -0.05, -0.5).finished();
    limits.delta_u_max = (VectorXd(2) <<  0.05,  0.5).finished();
    MPCController mpc_con(model, config, weights, limits);
    VectorXd u_con = mpc_con.solve(x0, x_ref.replicate(config.N, 1));
    EXPECT_LE(u_con(0),  0.05 + 1e-3);
    EXPECT_GE(u_con(0), -0.05 - 1e-3);
    EXPECT_LE(u_con(1),  0.5 + 1e-3);
    EXPECT_GE(u_con(1), -0.5 - 1e-3);
}

TEST_F(MPCControllerTest, RateCost) {
    VectorXd x0(4); x0 << 0, 0, 0, 5;
    VectorXd x_ref(4); x_ref << 0, 1, 1, 5;

    // Without rate cost — u_prev=0, so first du = u
    MPCController mpc_free(model, config, weights, limits);
    VectorXd u_free = mpc_free.solve(x0, x_ref.replicate(config.N, 1));

    // With high rate cost — first input should be smaller (smoother)
    weights.S = Vector2d(100.0, 100.0).asDiagonal();
    MPCController mpc_smooth(model, config, weights, limits);
    VectorXd u_smooth = mpc_smooth.solve(x0, x_ref.replicate(config.N, 1));

    EXPECT_LT(std::abs(u_smooth(0)), std::abs(u_free(0)));
    EXPECT_LT(std::abs(u_smooth(1)), std::abs(u_free(1)));
}

TEST_F(MPCControllerTest, StateConstraints) {
    limits.x_min = (VectorXd(4) << -1e10, 0.0, -1e10, -1e10).finished();
    limits.x_max = (VectorXd(4) <<  1e10, 2.0,  1e10,  1e10).finished();

    VectorXd x0(4); x0 << 0, 0.5, 0, 5;   // py=0.5, near lower boundary
    VectorXd x_ref(4); x_ref << 0, -1, 0, 5;  // reference below boundary

    MPCController mpc(model, config, weights, limits);
    mpc.solve(x0, x_ref.replicate(config.N, 1));

    // check py stays >= 0 across all horizon steps
    for (int k = 0; k < config.N; k++)
        EXPECT_GE(mpc.get_predicted_trajectory()(k, 1), -1e-3);
}
