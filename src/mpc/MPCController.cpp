#include "mpc/MPCController.hpp"
#include <cmath>
#include <OsqpEigen/OsqpEigen.h>
#include <chrono>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;

MPCController::MPCController(SystemModel& model, MPCConfig config) : model_(model), config_(config), solve_time_(0.0){ }

MatrixXd MPCController::get_predicted_trajectory() const {
    return predicted_trajectory_;
}

double MPCController::get_solve_time() const {
    return solve_time_;
}

VectorXd MPCController::solve(const VectorXd& x0, const VectorXd& x_ref) {
    int n = model_.state_dim();
    int m = model_.input_dim();
    int N = config_.N;

    // Linearization
    VectorXd u0 = VectorXd::Zero(m);
    MatrixXd A = model_.jacobian_x(x0, u0);
    MatrixXd B = model_.jacobian_u(x0, u0);

    // Discretization using Euler method
    MatrixXd A_d = MatrixXd::Identity(n, n) + config_.dt * A;
    MatrixXd B_d = config_.dt * B;


    // Build Sx (n*N x n) and Su (n*N x m*N)
    MatrixXd Sx = MatrixXd::Zero(n*N, n);
    MatrixXd Su = MatrixXd::Zero(n*N, m*N);

    MatrixXd A_pow = MatrixXd::Identity(n, n);
    for (int i = 0; i < N; i++) {
        A_pow = A_d * A_pow;
        Sx.block(i*n, 0, n, n) = A_pow;

        MatrixXd A_ij = MatrixXd::Identity(n, n);  // A^0 for j=i
        for (int j = i; j >= 0; j--) {
            Su.block(i*n, j*m, n, m) = A_ij * B_d;
            A_ij = A * A_ij;  // A^(i-j+1) for next iteration
        }
    }

    // Build Q_bar and R_bar
    MatrixXd Q_bar = MatrixXd::Zero(n*N, n*N);
    MatrixXd R_bar = MatrixXd::Zero(m*N, m*N);

    for (int i = 0; i < N; i++) {
        Q_bar.block(i*n, i*n, n, n) = config_.Q;
        R_bar.block(i*m, i*m, m, m) = config_.R;
    }

    // Build P and q
    MatrixXd P = Su.transpose() * Q_bar * Su + R_bar;
    VectorXd q = Su.transpose() * Q_bar * (Sx * x0 - x_ref.replicate(N, 1));

    // Convert to sparse
    SparseMatrix<double> P_sparse = P.sparseView();
    SparseMatrix<double> A_con = SparseMatrix<double>(0, m * N); // no constraints

    auto t_start = std::chrono::high_resolution_clock::now();   // Start timer
    // Setup solver
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.data()->setNumberOfVariables(m * N);
    solver.data()->setNumberOfConstraints(0);
    solver.data()->setHessianMatrix(P_sparse);
    solver.data()->setGradient(q);
    solver.initSolver();
    solver.solveProblem();
    
    auto t_end = std::chrono::high_resolution_clock::now();     // End timer
    solve_time_ = std::chrono::duration<double>(t_end - t_start).count();

    
    // Extract first control input
    VectorXd U = solver.getSolution();
    // Predicted trajectory
    predicted_trajectory_ = Sx * x0 + Su * U;
    
    return U.segment(0, m);
}

