#include "models/BicycleModel.hpp"
#include "mpc/MPCController.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main() {
    int T = 50; // simulation steps

    // Create system model and MPC controller
    double wheelbase = 2.;
    BicycleModel model(wheelbase);

    MPCConfig config;
    config.N = 30; // prediction horizon
    config.dt = 0.1; // time step
    config.Q = MatrixXd::Zero(4, 4);
    config.Q(1,1) = 10.0;  // py — lane tracking
    config.Q(2,2) = 50.0;  // psi — heading
    config.Q(3,3) = 10.0;  // v — velocity
    // px weight = 0, don't care about longitudinal position
    config.R = MatrixXd::Zero(2, 2);
    config.R(0,0) = 1.0;   // delta
    config.R(1,1) = 10.0; // a acceleration


    MPCController mpc(model, config);
    // Initial state and reference trajectory
    VectorXd x0(4);
    x0 << 0, 0.9, 0, 5; // [px, py, psi, v]

    VectorXd x_ref(4);
    x_ref << 0.0, 1.0, 0.0, 5.0;

    // Open CSV file for logging
    std::ofstream csv("output/trajectory.csv");
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV file\n";
        return 1;
    }
    csv << "t,px,py,psi,v,delta,a,solve_time\n";  // header


    VectorXd x = x0;
    for(int t = 0; t < T; t++) {
        // Solve MPC problem
        VectorXd u = mpc.solve(x, x_ref);

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
        << "," << u(0) << "," << u(1) << "," << mpc.get_solve_time() << "\n";

        // Propagate state
        x = x + config.dt * model.dynamics(x, u);
    }

    csv.close();



    return 0;
}