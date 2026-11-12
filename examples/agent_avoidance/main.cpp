// Moving-agent avoidance: an agent cuts into the ego's lane the SMPC yields laterally 
// with a margin that combines its own tube and the agent's growing prediction uncertainty.
//
// Two controllers are compared on the identical scenario:
//   - "deterministic prediction": agent variance = 0
//   - "chance constrained":       agent variance grows along the horizon

#include "models/BicycleModel.hpp"
#include "models/GPResidualModel.hpp"
#include "models/ConstantVelocityAgent.hpp"
#include "mpc/SMPCController.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <future>
#include <vector>
#include <algorithm>

using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector4d;

// The agent holds station beside the ego, then cuts in at a fixed step. Until then 
// the observed vy is 0, so constant-velocity extrapolation predicts a stationary agent
// over the whole horizon, that onset is what var_agent(k) = sigma0^2 + k*dt*q must cover.
struct AgentSim {
    double py       = 2.0;              // holding station, clear of the ego at py = 0
    double vy_cut   = -2.0;             // lateral speed once the cut-in starts
    double py_final = -0.3;             // settles here, forcing the ego to yield
    double vy_last  = 0.0;              // realized lateral speed of the last step
    int    steps    = 0;
    int    t_cut    = 20;               // step the cut-in begins: 2.0 s at dt = 0.1

    void step(double dt, std::mt19937& rng) {
        std::normal_distribution<double> agent_noise(0.0, 0.8);   // process noise on vy
        double vy = (steps >= t_cut) ? vy_cut + agent_noise(rng) : 0.0;
        double py_next = std::max(py + vy * dt, py_final);
        vy_last = (py_next - py) / dt;   // what an observer would measure
        py = py_next;
        steps++;
    }
};

struct SimResult {
    bool   violated   = false;   // ego ever came closer than r_safe to the agent
    double min_margin = 1e9;     // smallest (agent_py - r_safe - ego_py) over the run
};

// Both controllers on one seed - the unit of parallel work in the Monte-Carlo below
struct RunPair {
    SimResult det;
    SimResult chance;
};

// Runs one closed-loop episode. csv_path empty -> no logging (Monte-Carlo mode).
SimResult run_sim(SMPCController& controller,
                  ConstantVelocityAgent& agent,
                  const BicycleModel& bicycle,
                  const GPResidualModel& gp,
                  const MPCConfig& config,
                  const VectorXd& x0,
                  const VectorXd& x_ref,
                  double r_safe,
                  int T,
                  unsigned seed,
                  const std::string& csv_path) {

    SimResult result;
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "t,px,py,psi,v,delta,a,agent_py,clearance\n";
    }

    double k_true = 0.004;
    std::mt19937 rng(seed);

    AgentSim agent_sim;
    VectorXd x = x0;
    for (int t = 0; t < T; t++) {
        // All the controller gets: where the agent is now and how fast it last moved.
        double py_a = agent_sim.py;
        double vy_a = agent_sim.vy_last;
        agent.observe(py_a, vy_a);

        VectorXd u = controller.solve(x, x_ref.replicate(config.N, 1));

        // Lateral clearance: negative means the ego is inside the agent's safety radius.
        double margin = py_a - r_safe - x(1);
        result.min_margin = std::min(result.min_margin, margin);
        if (margin < 0.0) result.violated = true;

        if (csv.is_open()) {
            csv << t << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3)
                << "," << u(0) << "," << u(1)
                << "," << py_a << "," << (py_a - r_safe) << "\n";
        }

        // Plant: nominal bicycle + the true residual + GP-shaped noise on the heading rate
        VectorXd dx = bicycle.dynamics(x, u);
        dx(2) += -k_true * x(3) * x(3) * u(0);
        VectorXd x_next = x + config.dt * dx;

        double sigma = std::sqrt(gp.variance(x, u)(2));   // GP std dev on psi_dot, pre-update state
        std::normal_distribution<double> noise(0.0, sigma);
        x_next(2) += noise(rng);                          // same channel the SMPC's Sigma_w assumes

        x = x_next;
        agent_sim.step(config.dt, rng);                   // the agent's own cut-in
    }
    return result;
}

int main() {
    int T = 60;
    MPCConfig config;
    config.N = 20;
    config.dt = 0.1;

    double wheelbase = 2.7, r_safe = 1.0;
    BicycleModel bicycle(wheelbase);
    GPResidualModel gp(GP_PARAMS_PATH, wheelbase);

    MPCWeights weights(gp);
    weights.Q = Vector4d(0, 10, 50, 10).asDiagonal();
    weights.R = Vector2d(1, 10).asDiagonal();
    VectorXd x_trim = (VectorXd(4) << 0, 0, 0, 10).finished();
    weights.Qf = weights.compute_dare(gp, x_trim, config.dt);

    MPCLimits limits(gp);
    limits.u_min = Vector2d(-0.5, -5.0);
    limits.u_max = Vector2d( 0.5,  5.0);
    limits.x_min(1) = -2.5;
    limits.x_max(1) =  2.0;                    // lane bounds

    VectorXd x0(4);    x0    << 0, 0, 0, 10;   // centre of the lane, cruising
    VectorXd x_ref(4); x_ref << 0, 0, 0, 10;   // hold the centre

    // Run 1 - deterministic prediction: the agent's mean only, no prediction uncertainty
    {
        SMPCController smpc(gp, config, weights, limits, 0.95);
        ConstantVelocityAgent agent(0.0, 0.0);           // sigma0^2 = 0, q = 0
        smpc.set_agent(&agent, r_safe);
        run_sim(smpc, agent, bicycle, gp, config, x0, x_ref, r_safe, T, 0,
                              "output/agent_deterministic.csv");
    }

    // Run 2 - chance constrained: prediction uncertainty grows along the horizon
    {
        SMPCController smpc(gp, config, weights, limits, 0.95);
        ConstantVelocityAgent agent(/*sigma0_sq=*/0.01, /*q=*/0.1);
        smpc.set_agent(&agent, r_safe);
        run_sim(smpc, agent, bicycle, gp, config, x0, x_ref, r_safe, T, 0,
                              "output/agent_chance.csv");
    }

    std::cout << "wrote output/agent_deterministic.csv and output/agent_chance.csv\n";

    // Monte-Carlo: same scenario, many noise realizations. Both controllers get the
    // same seed per run, so any difference in violation rate comes from the constraint.
    int n_runs = 100;
    int viol_det = 0, viol_chance = 0;
    double worst_det = 1e9, worst_chance = 1e9;     // closest approach over all runs
    double sum_det   = 0.0, sum_chance   = 0.0;     // for the mean over runs

    // One seed = one unit of work: both controllers on the same disturbance realization.
    // Each task touches only local objects or const shared models, so runs are independent.
    auto run_seed = [&](unsigned s) {
        RunPair rp;
        {
            SMPCController smpc(gp, config, weights, limits, 0.95);
            ConstantVelocityAgent agent(0.0, 0.0);
            smpc.set_agent(&agent, r_safe);
            rp.det = run_sim(smpc, agent, bicycle, gp, config, x0, x_ref, r_safe, T, s, "");
        }
        {
            SMPCController smpc(gp, config, weights, limits, 0.95);
            ConstantVelocityAgent agent(0.01, 0.1);
            smpc.set_agent(&agent, r_safe);
            rp.chance = run_sim(smpc, agent, bicycle, gp, config, x0, x_ref, r_safe, T, s, "");
        }
        return rp;
    };

    std::cout << "\nMonte-Carlo: " << n_runs << " runs x 2 controllers (parallel)\n";

    // Launch run_seed in parallel and add to futured
    std::vector<std::future<RunPair>> futures;
    for (unsigned s = 0; s < static_cast<unsigned>(n_runs); s++)
        futures.push_back(std::async(std::launch::async, run_seed, s));

    // Output
    for (size_t i = 0; i < futures.size(); i++) {
        RunPair rp = futures[i].get();

        if (rp.det.violated) viol_det++;
        worst_det = std::min(worst_det, rp.det.min_margin);
        sum_det  += rp.det.min_margin;

        if (rp.chance.violated) viol_chance++;
        worst_chance = std::min(worst_chance, rp.chance.min_margin);
        sum_chance  += rp.chance.min_margin;

        std::cout << "  run " << (i + 1) << "/" << n_runs
                  << "  violations so far: det " << viol_det
                  << ", chance " << viol_chance << "\n";
    }

    std::cout << "\nMonte-Carlo over " << n_runs << " noise realizations"
              << "  (budget 1 - p = 5%)\n"
              << "  deterministic prediction : " << 100.0 * viol_det    / n_runs << "%"
              << "   min margin " << worst_det
              << " m, mean " << sum_det    / n_runs << " m\n"
              << "  chance constrained       : " << 100.0 * viol_chance / n_runs << "%"
              << "   min margin " << worst_chance
              << " m, mean " << sum_chance / n_runs << " m\n";
    return 0;
}
