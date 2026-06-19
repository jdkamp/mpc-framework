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


void run_sim(MPCController& controller,
            const std::function<VectorXd(const VectorXd&, const VectorXd&)>& plant,
            const VectorXd& x0,
            const std::function<VectorXd(int)>& x_ref_at,
            int T,
            const std::string& csv_path) {
    
    std::ofstream csv(csv_path);
    csv << "t,px,py,psi,v,delta,a,solve_time\n";

    VectorXd x = x0;
    for(int t = 0; t < T; t++) {
        VectorXd x_ref_traj = x_ref_at(t);
        VectorXd u = controller.solve(x, x_ref_traj);
        csv << t << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3)
            << "," << u(0) << "," << u(1) << "," << controller.get_solve_time() << "\n";

        x = plant(x, u);
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
    std::mt19937 rng(0);   // disturbance generator (re-seeded before each run for a fair comparison)
    auto plant = [&](const VectorXd& x, const VectorXd& u) -> VectorXd {
        VectorXd dx = bicycle.dynamics(x, u);
        dx(2) += -k_true * x(3) * x(3) * u(0) ;
        VectorXd x_next = x + config.dt * dx;

        double sigma = 1.0  * std::sqrt(gp.variance(x, u)(2));   // GP std-dev on heading rate
        std::normal_distribution<double> noise(0.0, sigma);
        x_next(2) += noise(rng);                          // stochastic disturbance (same Sigma_w the SMPC assumes)
        return x_next;
    };

    MPCWeights weights(bicycle);
    weights.Q = Vector4d(0.0, 1.0, 1.0, 1.0).asDiagonal();   // [px, py, psi, v]
    weights.R = Vector2d(20.0, 10.0).asDiagonal();               // [delta, a]
    weights.S = Vector2d(10.0, 10.0).asDiagonal();
    VectorXd x_trim = (VectorXd(4) << 0, 0, 0, v_ref).finished();
    weights.Qf = weights.compute_dare(bicycle, x_trim, config.dt);

    MPCLimits limits(bicycle);
    limits.u_min = (VectorXd(2) << -0.5, -5.0).finished();
    limits.u_max = (VectorXd(2) << 0.5, 5.0).finished();
    limits.delta_u_min = (VectorXd(2) << -1.0, -5.0).finished();
    limits.delta_u_max = (VectorXd(2) <<  1.0,  5.0).finished();

    limits.x_max(1) = 6.0;


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

    rng.seed(0); run_sim(detmpc, plant, x0, ref_at, T, "output/cmp_detmpc.csv");
    rng.seed(0); run_sim(gmpc,   plant, x0, ref_at, T, "output/cmp_gpmpc.csv");
    rng.seed(0); run_sim(smpc,   plant, x0, ref_at, T, "output/cmp_smpc.csv");

    std::cout << "Wrote output/cmp_{deterministic,gpmpc,gpsmpc}.csv\n";


    return 0;
}
