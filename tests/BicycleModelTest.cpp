#include <gtest/gtest.h>
#include "models/BicycleModel.hpp"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;

class BicycleModelTest : public ::testing::Test {
protected:
    BicycleModel model{2.7};
};

TEST_F(BicycleModelTest, DynamicsStraightLine) {
    VectorXd x(4);
    x << 0, 0, 0, 1;  // px, py, psi, vel

    VectorXd u(2);
    u << 0, 0;  // delta, a

    VectorXd xdot = model.dynamics(x, u);

    // Expect to move in x direction
    EXPECT_NEAR(xdot(0), 1, 1e-6);
    EXPECT_NEAR(xdot(1), 0, 1e-6);
    EXPECT_NEAR(xdot(2), 0, 1e-6);
    EXPECT_NEAR(xdot(3), 0, 1e-6);
}

TEST_F(BicycleModelTest, DynamicsCurveDriving)      {
    VectorXd x(4);
    x << 0, 0, M_PI/2, 1;  // px, py, psi, vel

    VectorXd u(2);
    u << 0, 0;  // delta, a

    VectorXd xdot = model.dynamics(x, u);

    // Expect to move in y direction
    EXPECT_NEAR(xdot(0), 0, 1e-6);
    EXPECT_NEAR(xdot(1), 1, 1e-6);
    EXPECT_NEAR(xdot(2), 0, 1e-6);
    EXPECT_NEAR(xdot(3), 0, 1e-6);
}

TEST_F(BicycleModelTest, StateDimension) {
    EXPECT_EQ(model.state_dim(), 4);
}

TEST_F(BicycleModelTest, InputDimension) {
    EXPECT_EQ(model.input_dim(), 2);
}

TEST_F(BicycleModelTest, JacobianDimension) {
    VectorXd x(4);
    x << 0, 0, 0, 1;  // px, py, psi, vel

    VectorXd u(2);
    u << 0, 0;  // delta, a

    // Check dimensions of Jacobians
    MatrixXd A = model.jacobian_x(x, u);
    EXPECT_EQ(A.rows(), 4);
    EXPECT_EQ(A.cols(), 4);

    MatrixXd B = model.jacobian_u(x, u);
    EXPECT_EQ(B.rows(), 4);
    EXPECT_EQ(B.cols(), 2);
}

TEST_F(BicycleModelTest, JacobianXValues) {
    VectorXd x(4);
    x << 0, 0, 0, 1;  // px, py, psi, vel

    VectorXd u(2);
    u << 0, 0;  // delta, a

    MatrixXd A = model.jacobian_x(x, u);

    // Check specific entries in the A matrix
    EXPECT_NEAR(A(0,2), 0, 1e-6);
    EXPECT_NEAR(A(0,3), 1, 1e-6);
    EXPECT_NEAR(A(1,2), 1, 1e-6);
    EXPECT_NEAR(A(1,3), 0, 1e-6);
    EXPECT_NEAR(A(2,3), 0, 1e-6);
}

TEST_F(BicycleModelTest, JacobianUValues) {
    VectorXd x(4);
    x << 0, 0, 0, 1;  // px, py, psi, vel

    VectorXd u(2);
    u << 0, 0;  // delta, a

    // Check specific entries in the B matrix
    MatrixXd B = model.jacobian_u(x, u);
    EXPECT_NEAR(B(2,0), 1/2.7, 1e-6);
    EXPECT_NEAR(B(3,1), 1, 1e-6);

}

TEST_F(BicycleModelTest, JacobianXNumerical) {
    VectorXd x(4); x << 0, 0, 0, 1;
    VectorXd u(2); u << 0, 0;
    double eps = 1e-5;

    MatrixXd A_analytical = model.jacobian_x(x, u);
    MatrixXd A_numerical = MatrixXd::Zero(4, 4);

    // Compute numerical Jacobian w.r.t. input u using central difference
    for (int j = 0; j < 4; j++) {
        VectorXd e = VectorXd::Zero(4);
        e(j) = 1.0;
        VectorXd diff = (model.dynamics(x + eps*e, u) - model.dynamics(x - eps*e, u));
        A_numerical.col(j) = diff / (2*eps);
    }

    // Compare analytical and numerical Jacobians
    EXPECT_TRUE(A_analytical.isApprox(A_numerical, 1e-4));
}


TEST_F(BicycleModelTest, JacobianUNumerical) {
    VectorXd x(4); x << 0, 0, 0, 1;
    VectorXd u(2); u << 0, 0;
    double eps = 1e-5;

    MatrixXd B_analytical = model.jacobian_u(x, u);
    MatrixXd B_numerical = MatrixXd::Zero(4, 2);

    // Compute numerical Jacobian w.r.t. input u using central difference
    for (int j = 0; j < 2; j++) {
        VectorXd e = VectorXd::Zero(2);
        e(j) = 1.0;
        VectorXd diff = (model.dynamics(x, u + eps*e) - model.dynamics(x, u - eps*e));
        B_numerical.col(j) = diff / (2*eps);
    }

    // Compare analytical and numerical Jacobians
    EXPECT_TRUE(B_analytical.isApprox(B_numerical, 1e-4));
}
