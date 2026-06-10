#include <gtest/gtest.h>
#include <models/GPResidualModel.hpp>
#include "models/BicycleModel.hpp"
#include <vector>

using Eigen::VectorXd;

class GPResidualModelTest : public ::testing::Test {
    protected:
    double wheelbase = 2.7;
    GPResidualModel gp_model{GP_PARAMS_PATH, wheelbase};
    BicycleModel bicycle_model{wheelbase};
};

TEST_F(GPResidualModelTest, MeanMatchesPython) {
    struct Case { double v, delta, mean; };
    std::vector<Case> cases {
        {5.0,  0.1, -0.011242},
        {10.0, 0.3, -0.121218},
        {3.0, -0.2,  0.007463},
        {8.0,  0.0, -0.001836},
    };
    for (auto c : cases) {
        VectorXd x(4); x << 0, 0, 0, c.v;
        VectorXd u(2); u << c.delta, 0;
        // GP correction = corrected dynamics - nominal dynamics, on psi_dot
        double correction = gp_model.dynamics(x, u)(2) - bicycle_model.dynamics(x, u)(2);
        EXPECT_NEAR(correction, c.mean, 1e-4);
    }
}

TEST_F(GPResidualModelTest, VarianceMatchesPython) {
    struct Case { double v, delta, var; };
    std::vector<Case> cases = {
        {5.0,  0.1, 0.000000},
        {10.0, 0.3, 0.000032},
        {3.0, -0.2, 0.000001},
        {8.0,  0.0, 0.000001},
    };
    for (auto c : cases) {
        VectorXd x(4); x << 0, 0, 0, c.v;
        VectorXd u(2); u << c.delta, 0;
        EXPECT_NEAR(gp_model.variance(x, u)(2), c.var, 1e-5);
    }
}

