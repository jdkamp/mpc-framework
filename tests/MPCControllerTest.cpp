#include <gtest/gtest.h>
#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class MPCControllerTest : public ::testing::Test {
protected:
    BicycleModel model{2.7};
    
    MPCConfig config;

    void SetUp() override {
        config.N = 30; // prediction horizon
        config.dt = 0.1; // time step
        config.Q = MatrixXd::Zero(4, 4);
        config.Q(0,0) = 10.0;  // px — longitudinal tracking
        config.Q(1,1) = 10.0;  // py — lane tracking
        config.Q(2,2) = 50.0;  // psi — heading
        config.Q(3,3) = 10.0;  // v — velocity
        config.Qf = config.Q;
        config.R = MatrixXd::Zero(2, 2);
        config.R(0,0) = 1.0;   // delta
        config.R(1,1) = 10.0; // a acceleration
        config.u_min = (VectorXd(2) << -1e10, -1e10).finished();  // no constraints
        config.u_max = (VectorXd(2) << 1e10, 1e10).finished();  // no constraints
        config.x_min = (VectorXd(4) << -1e10, -1e10, -1e10, -1e10).finished();  // no constraints
        config.x_max = (VectorXd(4) << 1e10, 1e10, 1e10, 1e10).finished();  // no constraints
    }
};

TEST_F(MPCControllerTest, NoControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref = x0;  // reference state is the same as initial state

    MPCController mpc(model, config);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_NEAR(u(0), 0, 1e-6);
    EXPECT_NEAR(u(1), 0, 1e-6);
}

TEST_F(MPCControllerTest, PxControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 100, 0, 0, 5 ;  // px, py, psi, vel

    MPCController mpc(model, config);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(1), 0);  // u(1) > 0

    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << -100, 0, 0, 0;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(1), 0);  // u(1) < 0
}

TEST_F(MPCControllerTest, PyControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 1, 0, 5;  // px, py, psi, vel

    MPCController mpc(model, config);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(0), 0);  // u(0) > 0

    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, -1, 0, 5;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(0), 0);  // u(0) < 0
}

TEST_F(MPCControllerTest, PsiControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 0, 1, 5;  // px, py, psi, vel

    MPCController mpc(model, config);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(0), 0);  // u(0) > 0

    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 0, -1, 5;  // px, py, psi, vel

    u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LT(u(0), 0);  // u(0) < 0
}

TEST_F(MPCControllerTest, VelControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << 0, 0, 0, 1;  // px, py, psi, vel

    MPCController mpc(model, config);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GT(u(1), 0);  // u(1) > 0

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

    MPCController mpc(model, config);
    VectorXd u = mpc.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_GE(u(0), 0.1);  // u(0) >= u_min(0)
    EXPECT_GE(u(1), 0.5);  // u(1) >= u_min(1)

    config.u_min = (VectorXd(2) << -0.1, -0.5).finished();  // max steering angle, max acceleration
    config.u_max = (VectorXd(2) << 0.1, 0.5).finished();  // min steering angle, min acceleration

    MPCController mpc_constrained(model, config);
    u = mpc_constrained.solve(x0,  x_ref.replicate(config.N, 1));

    EXPECT_LE(u(0), config.u_max(0) + 1e-3);  // u(0) <= u_max(0) + small tolerance
    EXPECT_LE(u(1), config.u_max(1) + 1e-3);  // u(1) <= u_max(1) + small tolerance
}

TEST_F(MPCControllerTest, StateConstraints) {
    config.x_min = (VectorXd(4) << -1e10, 0.0, -1e10, -1e10).finished();
    config.x_max = (VectorXd(4) <<  1e10, 2.0,  1e10,  1e10).finished();

    VectorXd x0(4); x0 << 0, 0.5, 0, 5;   // py=0.5, near lower boundary
    VectorXd x_ref(4); x_ref << 0, -1, 0, 5;  // reference below boundary

    MPCController mpc(model, config);
    mpc.solve(x0, x_ref.replicate(config.N, 1));

    // check py stays >= 0 across all horizon steps
    for (int k = 0; k < config.N; k++)
        EXPECT_GE(mpc.get_predicted_trajectory()(k, 1), -1e-3);
}
