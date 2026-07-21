#include <gtest/gtest.h>
#include <models/GPResidualModel.hpp>
#include "models/BicycleModel.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <fstream>

using Eigen::VectorXd;

class GPResidualModelTest : public ::testing::Test {
    protected:
    double wheelbase = 2.7;
    GPResidualModel gp_model{GP_PARAMS_PATH, wheelbase};
    BicycleModel bicycle_model{wheelbase};
    nlohmann::json params = nlohmann::json::parse(std::ifstream{GP_PARAMS_PATH});
};

TEST_F(GPResidualModelTest, MeanMatchesPython) {
    for (auto c : params["reference_checks"]) {
        double v = c["v"], delta = c["delta"], mean = c["mean"];
        VectorXd x(4); x << 0, 0, 0, v;
        VectorXd u(2); u << delta, 0;
        // GP correction = corrected dynamics - nominal dynamics, on psi_dot
        double correction = gp_model.dynamics(x, u)(2) - bicycle_model.dynamics(x, u)(2);
        EXPECT_NEAR(correction, mean, 1e-4);
    }
}

TEST_F(GPResidualModelTest, VarianceMatchesPython) {
    for (auto c : params["reference_checks"]) {
        double v = c["v"], delta = c["delta"], var = c["var"];
        VectorXd x(4); x << 0, 0, 0, v;
        VectorXd u(2); u << delta, 0;
        EXPECT_NEAR(gp_model.variance(x, u)(2), var, 1e-5);
    }
}

TEST_F(GPResidualModelTest, JacobianMatchesFiniteDifference) {
    struct Case { double v, delta; };
    std::vector<Case> cases { {5.0, 0.1}, {10.0, 0.3}, {3.0, -0.2}, {8.0, 0.05} };
    double eps = 1e-6;
    for (auto c : cases) {
        VectorXd x(4); x << 0, 0, 0, c.v;
        VectorXd u(2); u << c.delta, 0;

        VectorXd xp = x; xp(3) += eps;
        double fd_v = (gp_model.dynamics(xp, u)(2) - gp_model.dynamics(x, u)(2)) / eps;
        EXPECT_NEAR(gp_model.jacobian_x(x, u)(2, 3), fd_v, 1e-4);

        VectorXd up = u; up(0) += eps;
        double fd_d = (gp_model.dynamics(x, up)(2) - gp_model.dynamics(x, u)(2)) / eps;
        EXPECT_NEAR(gp_model.jacobian_u(x, u)(2, 0), fd_d, 1e-4);
    }
}
