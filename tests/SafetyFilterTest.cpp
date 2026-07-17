#include <gtest/gtest.h>
#include "mpc/SafetyFilter.hpp"
#include "models/GPResidualModel.hpp"
#include "models/BicycleModel.hpp"
#include <random>
#include <algorithm>

using Eigen::VectorXd;

TEST(SafetyFilterTest, PassThroughSafeAction) {
    double wheelbase = 2.7;
    GPResidualModel gp(GP_PARAMS_PATH, wheelbase);
    MPCConfig config;
    config.N = 30;
    config.dt = 0.1;
    MPCWeights weights(gp);
    MPCLimits limits(gp);
    limits.u_min = (VectorXd(2) << -0.5, -3.0).finished();
    limits.u_max = (VectorXd(2) <<  0.5,  3.0).finished();
    limits.x_max(1) = 9.0;

    SafetyFilter filter(gp, config, weights, limits, 0.95);

    VectorXd x0(4); x0 << 0.0, 0.0, 0.0, 10.0;
    VectorXd u_rl(2); u_rl << 0.1, 0.0;
    VectorXd u = filter.filter(x0, u_rl);

    EXPECT_NEAR(u(0), u_rl(0), 1e-4);
    EXPECT_NEAR(u(1), u_rl(1), 1e-4);
}

TEST(SafetyFilterTest, BlocksUnsafeAction) {
    double wheelbase = 2.7;
    GPResidualModel gp(GP_PARAMS_PATH, wheelbase);
    MPCConfig config;
    config.N = 30;
    config.dt = 0.1;
    MPCWeights weights(gp);
    MPCLimits limits(gp);
    limits.u_min = (VectorXd(2) << -0.5, -3.0).finished();
    limits.u_max = (VectorXd(2) <<  0.5,  3.0).finished();
    limits.x_max(1) = 9.0;

    SafetyFilter filter(gp, config, weights, limits, 0.95);

    VectorXd x0(4); x0 << 0.0, 8.0, 0.3, 10.0;  // near limit 0, heading into it
    VectorXd u_rl(2); u_rl << 0.4, 0.0;         // agent: steer into limit
    VectorXd u = filter.filter(x0, u_rl);

    EXPECT_LT(u(0), u_rl(0));   // must reduce the steering
    EXPECT_LT(u(0), 0.1);
}

TEST(SafetyFilterTest, KeepsUnsafeAgent) {
    double wheelbase = 2.7;
    GPResidualModel gp(GP_PARAMS_PATH, wheelbase);
    BicycleModel bicyle(wheelbase);
    double k_true = 0.004;
    MPCConfig config;
    config.N = 30;
    config.dt = 0.1;
    MPCWeights weights(gp);
    MPCLimits limits(gp);
    limits.u_min = (VectorXd(2) << -0.5, -3.0).finished();
    limits.u_max = (VectorXd(2) <<  0.5,  3.0).finished();
    limits.x_max(1) = 9.0;

    SafetyFilter filter(gp, config, weights, limits, 0.95);

    VectorXd x(4); x << 0.0, 0.0, 0.0, 10.0;
    std::mt19937 rng(0);
    double max_py = 0.0;

    for(int t = 0; t < 50; t++) {
        // the unsafe PD agent, resurrected: goal beyond the bound
        double delta = std::clamp(0.1 * (12.0 - x(1)) - 1.0 * x(2), -0.5, 0.5);
        VectorXd u = filter.filter(x, (VectorXd(2) << delta, 0.0).finished());
        
        VectorXd dx = bicyle.dynamics(x, u);
        dx(2) += -k_true * x(3) * x(3) * u(0);
        x = x + config.dt * dx;
        double sigma = std::sqrt(gp.variance(x, u)(2));
        std::normal_distribution<double> noise(0.0, sigma);
        x(2) += noise(rng);

        max_py = std::max(max_py, x(1));
    }

    EXPECT_LT(max_py, 9.0);     // never exceedes
    EXPECT_GT(max_py, 8.0);     // and went near limit
}