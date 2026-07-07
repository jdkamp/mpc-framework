#include "models/BicycleModel.hpp"
#include "models/GPResidualModel.hpp"
#include "mpc/MPCController.hpp"
#include "mpc/SMPCController.hpp"
#include <iostream>
#include <fstream>
#include <functional>
#include <cmath>
#include <random>

using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector4d;


int run_noisy_sim(MPCController& controller,
            const std::function<VectorXd(const VectorXd&, const VectorXd&, std::mt19937&)>& plant,
            const VectorXd& x0,
            const std::function<VectorXd(int)>& x_ref_at,
            int T,
            double py_bound,
            std::mt19937& rng,
            double& max_py) {
    
    int violations = 0;
    VectorXd x = x0;
    for(int t = 0; t < T; t++) {
        VectorXd u = controller.solve(x, x_ref_at(t));
        x = plant(x, u, rng);
        if (x(1) > max_py) max_py = x(1);
        if(x(1) > py_bound) violations++;
    }
    return violations;
}


// Log N_show noisy py-trajectories for one controller (for the Monte-Carlo fan plot)
void log_ensemble(MPCController& controller,
            const std::function<VectorXd(const VectorXd&, const VectorXd&, std::mt19937&)>& plant,
            const VectorXd& x0,
            const std::function<VectorXd(int)>& x_ref_at,
            int T, int N_show, const std::string& csv_path) {
    std::ofstream csv(csv_path);
    csv << "rollout,t,py\n";
    for (int i = 0; i < N_show; i++) {
        std::mt19937 rng(i);
        VectorXd x = x0;
        for (int t = 0; t < T; t++) {
            VectorXd u = controller.solve(x, x_ref_at(t));
            csv << i << "," << t << "," << x(1) << "\n";   // py per (rollout, t)
            x = plant(x, u, rng);
        }
    }
}


int main() {
    int T = 50;
    MPCConfig config;
    config.N = 30;
    config.dt = 0.1;

    double v_ref = 10.0;
    double R = 8.0;

    double wheelbase = 2.7;
    BicycleModel bicycle(wheelbase);
    GPResidualModel gp(GP_PARAMS_PATH, wheelbase);

    double k_true = 0.004;
    auto plant = [&](const VectorXd& x, const VectorXd& u, std::mt19937& rng) -> VectorXd {
        VectorXd dx = bicycle.dynamics(x, u);
        dx(2) += -k_true * x(3) * x(3) * u(0) ;
        VectorXd x_next =  x + config.dt * dx;

        double sigma = std::sqrt(gp.variance(x, u)(2));
        std::normal_distribution<double> noise(0.0, sigma);
        x_next(2) += noise(rng);

        return x_next;
    };

    MPCWeights weights(bicycle);
    weights.Q = Vector4d(0.0, 100.0, 100.0, 100.0).asDiagonal();   // [px, py, psi, v]
    weights.R = Vector2d(5.0, 5.0).asDiagonal();               // [delta, a]
    weights.S = Vector2d(1.0, 1.0).asDiagonal();
    VectorXd x_trim = (VectorXd(4) << 0, 0, 0, v_ref).finished();
    weights.Qf = weights.compute_dare(bicycle, x_trim, config.dt);

    MPCLimits limits(bicycle);
    limits.u_min = (VectorXd(2) << -0.5, -5.0).finished();
    limits.u_max = (VectorXd(2) << 0.5, 5.0).finished();
    limits.delta_u_min = (VectorXd(2) << -1.0, -5.0).finished();
    limits.delta_u_max = (VectorXd(2) <<  1.0,  5.0).finished();
    
    double py_bound = 9.0;
    limits.x_max(1) = py_bound;


    VectorXd x0(4);       x0 << 0, 0, 0, v_ref; // start at v_ref
    auto ref_at = [&](int t) -> VectorXd {
        VectorXd traj(4 * config.N);
        for (int k = 0; k < config.N; k++) {
            double s     = v_ref * config.dt * (t + k + 1);  // arc length ahead
            double psi_k = s / R;
            double px_k  = R * std::sin(psi_k);
            double py_k  = R * (1.0 - std::cos(psi_k));       // left turn, center (0,R)
            traj.segment(4 * k, 4) << px_k, py_k, psi_k, v_ref;
        }
        return traj;
    };


    double p = 0.95;
    MPCController detmpc(bicycle, config, weights, limits);
    MPCController gmpc(gp, config, weights, limits);
    SMPCController smpc(gp, config, weights, limits, p);

    int N_mc = 200;

    double max_det = 0, max_gp = 0, max_smpc = 0;
    int total_steps = N_mc * T;

    int viol_det = 0;
    for(int i = 0; i < N_mc; i++) {
        std::mt19937 rng(i); // different seed for each simulation
        viol_det += run_noisy_sim(detmpc, plant, x0, ref_at, T, py_bound, rng, max_det);

    }
    
    int viol_gp = 0;
    for(int i = 0; i < N_mc; i++) {
        std::mt19937 rng(i); // different seed for each simulation
        viol_gp += run_noisy_sim(gmpc, plant, x0, ref_at, T, py_bound, rng, max_gp);
        
    }
    
    int viol_smpc = 0;
    for(int i = 0; i < N_mc; i++) {
        std::mt19937 rng(i); // different seed for each simulation
        viol_smpc += run_noisy_sim(smpc, plant, x0, ref_at, T, py_bound, rng, max_smpc);
        
    }
    
    std::cout << "Deterministic MPC violations: "   << viol_det     << "\n";
    std::cout << "GP MPC violations: "              << viol_gp      << "\n";
    std::cout << "SMPC violations: "                << viol_smpc    << "\n";
    std::cout << "peak py — deterministic:"  << max_det << "  gp:" << max_gp << "  smpc:" << max_smpc << "\n";


    std::ofstream csv("output/mc_violations.csv");
    csv << "controller,violations,total,rate_pct\n";
    csv << "Deterministic MPC," << viol_det  << "," << total_steps << "," << 100.0*viol_det / total_steps  << "\n";
    csv << "GP-MPC,"            << viol_gp   << "," << total_steps << "," << 100.0*viol_gp / total_steps   << "\n";
    csv << "SMPC,"              << viol_smpc << "," << total_steps << "," << 100.0*viol_smpc / total_steps << "\n";

    csv.close();

    // Log a few rollouts per controller for the fan plot
    int N_show = 40;
    log_ensemble(detmpc, plant, x0, ref_at, T, N_show, "output/ens_detmpc.csv");
    log_ensemble(gmpc,   plant, x0, ref_at, T, N_show, "output/ens_gpmpc.csv");
    log_ensemble(smpc,   plant, x0, ref_at, T, N_show, "output/ens_smpc.csv");
    std::cout << "Wrote output/ens_{detmpc,gpmpc,smpc}.csv\n";

    return 0;
}
