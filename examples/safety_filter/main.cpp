#include "models/BicycleModel.hpp"
#include "models/GPResidualModel.hpp"
#include "mpc/SafetyFilter.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector4d;

int main() {
    int T = 50;
    MPCConfig config;
    config.N = 30;
    config.dt = 0.1;

    double wheelbase = 2.7;
    BicycleModel bicycle(wheelbase);
    GPResidualModel gp(GP_PARAMS_PATH, wheelbase);

    // The true plant: nominal bicycle + true residual + GP-sigma disturbance
    double k_true = 0.004;
    std::mt19937 rng(0);
    auto plant = [&](const VectorXd& x, const VectorXd& u) -> VectorXd {
        VectorXd dx = bicycle.dynamics(x, u);
        dx(2) += -k_true * x(3) * x(3) * u(0);
        VectorXd x_next = x + config.dt * dx;
        double sigma = std::sqrt(gp.variance(x, u)(2));
        std::normal_distribution<double> noise(0.0, sigma);
        x_next(2) += noise(rng);
        return x_next;
    };

    // The unsafe agent: PD toward a goal beyond the bound
    double py_goal = 12.0;
    auto agent = [&](const VectorXd& x) -> VectorXd {
        double delta = std::clamp(0.1 * (py_goal - x(1)) - 1.0 * x(2), -0.5, 0.5);
        return (VectorXd(2) << delta, 0.0).finished();
    };

    MPCWeights weights(gp);
    MPCLimits limits(gp);
    limits.u_min = (VectorXd(2) << -0.5, -3.0).finished();
    limits.u_max = (VectorXd(2) <<  0.5,  3.0).finished();
    limits.delta_u_min = (VectorXd(2) << -1.0, -5.0).finished();
    limits.delta_u_max = (VectorXd(2) <<  1.0,  5.0).finished();
    limits.x_max(1) = 9.0;

    SafetyFilter filter(gp, config, weights, limits, 0.95);

    VectorXd x0(4); x0 << 0.0, 0.0, 0.0, 10.0;

    // Run 1: the agent drives the plant unfiltered
    rng.seed(0);
    VectorXd x = x0;
    std::ofstream raw("output/sf_raw.csv");
    raw << "t,py,delta_rl\n";
    for(int t = 0; t < T; t++) {
        VectorXd u_rl = agent(x);
        raw << t << "," << x(1) << "," << u_rl(0) << "\n";
        x = plant(x, u_rl);
    }

    // Run 2: filtered: same seed, same agent, filter in between
    rng.seed(0);
    x = x0;
    std::ofstream flt("output/sf_filtered.csv");
    flt << "t,py,delta_rl,delta_filtered\n";
    for (int t = 0; t < T; t++) {
        VectorXd u_rl = agent(x);
        VectorXd u = filter.filter(x, u_rl);
        flt << t << "," << x(1) << "," << u_rl(0) << "," << u(0) << "\n";
        x = plant(x, u);
    }

    std::cout << "Wrote output/sf_{raw,filtered}.csv\n";
    return 0;
}