#include <gtest/gtest.h>
#include "mpc/SafetyFilter.hpp"
#include "models/GPResidualModel.hpp"
#include "models/BicycleModel.hpp"
#include "models/ConstantVelocityAgent.hpp"
#include <random>
#include <algorithm>

using Eigen::VectorXd;
using namespace mpc;

class SafetyFilterTest : public ::testing::Test {
protected:
    double wheelbase = 2.7;
    GPResidualModel gp{GP_PARAMS_PATH, wheelbase};
    BicycleModel bicycle{wheelbase};
    double k_true = 0.004;
    MPCConfig config;
    MPCWeights weights{gp};
    MPCLimits limits{gp};

    void SetUp() override {
        config.N = 30;
        config.dt = 0.1;
        limits.u_min = (VectorXd(2) << -0.5, -3.0).finished();
        limits.u_max = (VectorXd(2) <<  0.5,  3.0).finished();
        limits.x_max(1) = 9.0;
    }

    // The unsafe policy both closed-loop tests share: a PD driving at a goal beyond the
    // bound. Returns the largest py reached. noise = false gives a deterministic run.
    double run_unsafe_policy(SafetyFilter& filter, int steps, bool noise) {
        VectorXd x(4); x << 0.0, 0.0, 0.0, 10.0;
        std::mt19937 rng(0);
        // Portable Gaussian: std::normal_distribution differs across standard libraries,
        // only std::mt19937 itself is specified, so CI would disagree with local runs.
        auto uniform01 = [&rng]() {
            return (rng() - std::mt19937::min()) /
                   (static_cast<double>(std::mt19937::max()) - std::mt19937::min() + 1.0);
        };

        double max_py = 0.0;
        for (int t = 0; t < steps; t++) {
            double delta = std::clamp(0.1 * (12.0 - x(1)) - 1.0 * x(2), -0.5, 0.5);
            VectorXd u = filter.filter(x, (VectorXd(2) << delta, 0.0).finished()).u;

            VectorXd dx = bicycle.dynamics(x, u);
            dx(2) += -k_true * x(3) * x(3) * u(0);
            x = x + config.dt * dx;

            if (noise) {
                double sigma = std::sqrt(gp.variance(x, u)(2));
                double u1 = uniform01(), u2 = uniform01();
                double z = std::sqrt(-2.0 * std::log(u1 + 1e-12)) * std::cos(2.0 * M_PI * u2);
                x(2) += sigma * z;
            }

            max_py = std::max(max_py, x(1));
        }
        return max_py;
    }
};

// A safe proposal must pass through untouched
TEST_F(SafetyFilterTest, PassThroughSafeAction) {
    SafetyFilter filter(gp, config, weights, limits, 0.95);

    VectorXd x0(4); x0 << 0.0, 0.0, 0.0, 10.0;
    VectorXd u_rl(2); u_rl << 0.1, 0.0;
    VectorXd u = filter.filter(x0, u_rl).u;

    EXPECT_NEAR(u(0), u_rl(0), 1e-4);
    EXPECT_NEAR(u(1), u_rl(1), 1e-4);
}

// Near the bound and heading into it, the proposal must be overridden
TEST_F(SafetyFilterTest, BlocksUnsafeAction) {
    SafetyFilter filter(gp, config, weights, limits, 0.95);

    VectorXd x0(4); x0 << 0.0, 8.0, 0.3, 10.0;  // near limit 0, heading into it
    VectorXd u_rl(2); u_rl << 0.4, 0.0;         // agent: steer into limit
    VectorXd u = filter.filter(x0, u_rl).u;

    EXPECT_LT(u(0), u_rl(0));   // must reduce the steering
    EXPECT_LT(u(0), 0.1);
}

// Closed loop under disturbance: the filter holds the lane bound against a policy that
// spends every step trying to cross it
TEST_F(SafetyFilterTest, KeepsUnsafeAgent) {
    SafetyFilter filter(gp, config, weights, limits, 0.95);

    double max_py = run_unsafe_policy(filter, 50, /*noise=*/true);

    EXPECT_LT(max_py, 9.0);     // never exceedes
    EXPECT_GT(max_py, 8.0);     // and went near limit
}

// SafetyFilter has no agent-avoidance code of its own: it inherits set_agent() from
// SMPCController. Same filter, same policy, same limits as KeepsUnsafeAgent - which
// reaches py ~ 8.9 - so stopping below 2.0 can only come from the agent constraint.
TEST_F(SafetyFilterTest, InheritsAgentConstraint) {
    SafetyFilter filter(gp, config, weights, limits, 0.95);

    double r_safe = 1.0;
    ConstantVelocityAgent agent(0.01, 0.0);     // constant variance, no growth
    agent.observe(3.0, 0.0);                    // parked at py = 3 -> boundary at 2.0
    filter.set_agent(&agent, r_safe);

    double max_py = run_unsafe_policy(filter, 50, /*noise=*/false);

    EXPECT_LT(max_py, 3.0 - r_safe);   // never enters the safety radius
    EXPECT_GT(max_py, 1.0);            // and did approach it - not just clamped to 0
}
