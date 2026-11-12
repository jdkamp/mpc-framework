#include <gtest/gtest.h>
#include "models/ConstantVelocityAgent.hpp"
using namespace mpc;

TEST(AgentPredictorTest, ConstantVelocityTest) {
    int N = 10;
    double dt = 0.1, sigma0_sq = 0.01,  q = 0.1, py = 1.0, vy = 0.5;
    ConstantVelocityAgent agent(sigma0_sq, q);
    agent.observe(py, vy);
    auto pred = agent.predict(N, dt);
    ASSERT_NEAR(pred.mean(0), py + dt * vy, 1e-9);                  // look one step ahead
    ASSERT_NEAR(pred.variance(0), sigma0_sq + dt * q, 1e-9);
    for(int i = 1; i < N; i++) {
        ASSERT_NEAR(pred.mean(i) - pred.mean(i-1), dt * vy, 1e-9);  // mean = py + dt * vy
        ASSERT_GT(pred.variance(i), pred.variance(i-1));            // variance grows
    }
}