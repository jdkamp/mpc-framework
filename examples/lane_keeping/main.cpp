#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector4d;


int main() {
    int T = 50; // simulation steps

    // Create system model and MPC controller
    double wheelbase = 2.;
    BicycleModel model(wheelbase);

    MPCConfig config;
    MPCWeights weights(model);
    MPCLimits limits(model);
    config.N = 30;                                              // prediction horizon
    config.dt = 0.1;                                            // time step
    weights.Q = Vector4d(0.0, 10.0, 50.0, 10.0).asDiagonal();    // [px, py, psi, v]
    VectorXd x_trim = (VectorXd(4) << 0, 0, 0, 5.0).finished(); // trim at straight driving, v=5
    weights.Qf = weights.compute_dare(model, x_trim, config.dt);
    weights.R = Vector2d(1.0, 10.0).asDiagonal();                // [delta, a]
    weights.S = Vector2d(10.0, 10.0).asDiagonal();             // [delta change rate, acceleration change rate]
    limits.u_min = (VectorXd(2) << -0.5, -3.0).finished();      // max steering angle, max acceleration
    limits.u_max = (VectorXd(2) << 0.5, 3.0).finished();        // min steering angle, min acceleration
    limits.x_min = (VectorXd(4) << -1e10, 0.0, -1e10, -1e10).finished(); // min state
    limits.x_max = (VectorXd(4) << 1e10,  2.0,  1e10,  1e10).finished(); // max state
    limits.delta_u_min = (VectorXd(2) << -1e10, -1e10).finished();  // no constraints
    limits.delta_u_max = (VectorXd(2) << 1e10, 1e10).finished();  // no constraints
    
    MPCController mpc(model, config, weights, limits);

    // Initial state and reference trajectory
    VectorXd x0(4);
    x0 << 0, 0.9, 0, 5; // [px, py, psi, v]

    VectorXd x_ref(4);
    x_ref << 0.0, 1.0, 0.0, 5.0;
    VectorXd x_ref_traj = x_ref.replicate(config.N, 1); // reference trajectory for N steps

    // Open CSV file for logging
    std::ofstream csv("output/trajectory.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV file\n";
        return 1;
    }
    csv << "t,px,py,psi,v,delta,a,solve_time,px_ref,py_ref,psi_ref,v_ref\n";  // header


    VectorXd x = x0;
    for(int t = 0; t < T; t++) {
        // Solve MPC problem
        VectorXd u = mpc.solve(x, x_ref_traj);

        // Print state and input
        std::cout << std::fixed << std::setprecision(4)
                << "t=" << std::setw(3) << t << " x=";
        for (int i = 0; i < x.size(); i++)
            std::cout << std::setw(9) << x(i);
        std::cout << " u=";
        for (int i = 0; i < u.size(); i++)
            std::cout << std::setw(9) << u(i);
        std::cout << " time=" << std::setprecision(6) << mpc.get_solve_time() << "\n";

        // Log to CSV
        csv << t << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3)
        << "," << u(0) << "," << u(1) << "," << mpc.get_solve_time()
        << "," << x_ref(0) << "," << x_ref(1) << "," << x_ref(2) << "," << x_ref(3) << "\n";

        // Propagate state
        x = x + config.dt * model.dynamics(x, u);
    }

    csv.close();



    return 0;
}