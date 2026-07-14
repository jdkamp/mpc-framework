#include "models/BicycleModel.hpp"
#include <fstream>

int main() {
    BicycleModel bicycle(2.7);
    double dt = 0.1;
    Eigen::VectorXd x(4); x << 0.0, 0.5, 0.2, 10.0;
    Eigen::VectorXd u(2); u << 0.05, 0.2;

    std::ofstream csv("data/parity_reference.csv");
    csv << "px,py,psi,v\n" << std::setprecision(17);
    for (int t = 0; t < 100; t++) {
        csv << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "\n";
        x = x + dt * bicycle.dynamics(x, u);
    }
}