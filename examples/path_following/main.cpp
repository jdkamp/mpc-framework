#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector4d;


int main() {
    int T = 500; // simulation steps
    
    // Path parameters
    double A = 1.0; // amplitude
    double lambda = 20.0; // wavelength
    double v_ref = 5.0; // reference velocity

    // Create system model and MPC controller
    double wheelbase = 2.7;
    BicycleModel model(wheelbase);
    MPCConfig config;
    config.N = 30;                                              // prediction horizon
    config.dt = 0.1;                                            // time step
    config.Q = Vector4d(0.0, 10.0, 50.0, 10.0).asDiagonal();    // [px, py, psi, v]
    config.Qf = Vector4d(0.0, 20.0, 100.0, 20.0).asDiagonal();  // [px, py, psi, v]
    config.R = Vector2d(1.0, 10.0).asDiagonal();                // [delta, a]
    config.u_min = (VectorXd(2) << -0.5, -3.0).finished();      // max steering angle, max acceleration
    config.u_max = (VectorXd(2) << 0.5, 3.0).finished();        // min steering angle, min acceleration

    MPCController mpc(model, config);

    // Initial state
    VectorXd x0(4);
    x0 << 0, 0, 0, 0; // [px, py, psi, v]
    VectorXd x = x0;

     // Open CSV file for logging
    std::ofstream csv("output/trajectory.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV file\n";
        return 1;
    }
    csv << "t,px,py,psi,v,delta,a,solve_time,px_ref,py_ref,psi_ref,v_ref\n";  // header

    for(int t = 0; t < T; t++) {
        // Reference state on the path
        VectorXd x_ref_traj(4*config.N);
        for (int k = 0; k < config.N; k++) {
            double px_k = x(0) + (k+1) * v_ref * config.dt;
            double py_k = A * std::sin(2 * M_PI * px_k / lambda);
            double psi_k = std::atan2(2 * M_PI * A / lambda * std::cos(2 * M_PI * px_k / lambda), 1);
            x_ref_traj.segment<4>(4*k) << px_k, py_k, psi_k, v_ref;
        }
        

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
        << "," << x_ref_traj(0) << "," << x_ref_traj(1) << "," << x_ref_traj(2) << "," << x_ref_traj(3) << "\n";

        // Propagate state
        x = x + config.dt * model.dynamics(x, u);
    }

    csv.close();

    return 0;
}

