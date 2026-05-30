#include <gtest/gtest.h>
#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class MPCControllerTest : public ::testing::Test {
protected:
    BicycleModel model{2.7};
    
    MPCConfig config;
    std::unique_ptr<MPCController> mpc;

    void SetUp() override {
        config.N = 30; // prediction horizon
        config.dt = 0.1; // time step
        config.Q = MatrixXd::Zero(4, 4);
        config.Q(0,0) = 10.0;  // px — longitudinal tracking
        config.Q(1,1) = 10.0;  // py — lane tracking
        config.Q(2,2) = 50.0;  // psi — heading
        config.Q(3,3) = 10.0;  // v — velocity
        config.R = MatrixXd::Zero(2, 2);
        config.R(0,0) = 1.0;   // delta
        config.R(1,1) = 10.0; // a acceleration
        config.u_min = (VectorXd(2) << -0.5, -3.0).finished();  // max steering angle, max acceleration
        config.u_max = (VectorXd(2) << 0.5, 3.0).finished();    // min steering angle, min acceleration
    
        mpc = std::make_unique<MPCController>(model, config);
    }
};

TEST_F(MPCControllerTest, NoControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref = x0;  // reference state is the same as initial state

    VectorXd u = mpc->solve(x0, x_ref);

    EXPECT_NEAR(u(0), 0, 1e-6);
    EXPECT_NEAR(u(1), 0, 1e-6);
}

TEST_F(MPCControllerTest, PxControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 1, 0, 0, 5 ;  // px, py, psi, vel

    VectorXd u = mpc->solve(x0, x_ref);

    EXPECT_GT(u(1), 0);  // u(1) > 0

    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << -1, 0, 0, 0;  // px, py, psi, vel

    u = mpc->solve(x0, x_ref);

    EXPECT_LT(u(1), 0);  // u(1) < 0
}

TEST_F(MPCControllerTest, PyControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 1, 0, 5;  // px, py, psi, vel

    VectorXd u = mpc->solve(x0, x_ref);

    EXPECT_GT(u(0), 0);  // u(0) > 0

    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, -1, 0, 5;  // px, py, psi, vel

    u = mpc->solve(x0, x_ref);

    EXPECT_LT(u(0), 0);  // u(0) < 0
}

TEST_F(MPCControllerTest, PsiControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 0, 1, 5;  // px, py, psi, vel

    VectorXd u = mpc->solve(x0, x_ref);

    EXPECT_GT(u(0), 0);  // u(0) > 0

    x0 << 0, 0, 0, 5;  // px, py, psi, vel
    x_ref << 0, 0, -1, 5;  // px, py, psi, vel

    u = mpc->solve(x0, x_ref);

    EXPECT_LT(u(0), 0);  // u(0) < 0
}

TEST_F(MPCControllerTest, VelControlDeviation) {
    VectorXd x0(4);
    VectorXd x_ref(4);
    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << 0, 0, 0, 1;  // px, py, psi, vel

    VectorXd u = mpc->solve(x0, x_ref);

    EXPECT_GT(u(1), 0);  // u(1) > 0

    x0 << 0, 0, 0, 0;  // px, py, psi, vel
    x_ref << 0, 0, 0, -1;  // px, py, psi, vel

    u = mpc->solve(x0, x_ref);

    EXPECT_LT(u(1), 0);  // u(1) < 0
}