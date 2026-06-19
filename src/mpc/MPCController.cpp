#include "mpc/MPCController.hpp"
#include <cmath>
#include <OsqpEigen/OsqpEigen.h>
#include <chrono>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::SparseMatrix;

// MPCWeights
MPCWeights::MPCWeights(const SystemModel& model) : // default to identity weights
    Q(MatrixXd::Identity(model.state_dim(), model.state_dim())),
    Qf(MatrixXd::Identity(model.state_dim(), model.state_dim())),
    R(MatrixXd::Identity(model.input_dim(), model.input_dim())),
    S(MatrixXd::Zero(model.input_dim(), model.input_dim()))
{}

MatrixXd MPCWeights::compute_dare(
    const SystemModel& model,
    const VectorXd& x_trim,
    const VectorXd& u_trim,
    double dt,
    int max_iter,
    double tol) const
{
    int n = model.state_dim();
    MatrixXd A_d = MatrixXd::Identity(n, n) + dt * model.jacobian_x(x_trim, u_trim);
    MatrixXd B_d = dt * model.jacobian_u(x_trim, u_trim);
    MatrixXd P = Q;
    for (int i = 0; i < max_iter; i++) {
        MatrixXd M = R + B_d.transpose() * P * B_d;
        MatrixXd S = M.inverse();
        MatrixXd P_new = Q + A_d.transpose() * P * A_d
            - A_d.transpose() * P * B_d * S * B_d.transpose() * P * A_d;
        if ((P_new - P).norm() < tol) return P_new;
        P = P_new;
    }
    return P;
}

MatrixXd MPCWeights::compute_dare(
    const SystemModel& model,
    const VectorXd& x_trim,
    double dt,
    int max_iter,
    double tol) const
{
    return compute_dare(model, x_trim, VectorXd::Zero(model.input_dim()), dt, max_iter, tol);
}

MatrixXd MPCWeights::compute_lqr_gain(
    const SystemModel& model,
    const VectorXd& x_trim,
    const VectorXd& u_trim,
    double dt,
    int max_iter,
    double tol) const
{
    int n = model.state_dim();
    MatrixXd A_d = MatrixXd::Identity(n, n) + dt * model.jacobian_x(x_trim, u_trim);
    MatrixXd B_d = dt * model.jacobian_u(x_trim, u_trim);
    MatrixXd P = compute_dare(model, x_trim, u_trim, dt, max_iter, tol);    // reuse DARE
    MatrixXd M = R + B_d.transpose() * P * B_d;
    return -M.inverse() * B_d.transpose() * P * A_d;                        // K
}

MatrixXd MPCWeights::compute_lqr_gain(
    const SystemModel& model,
    const VectorXd& x_trim,
    double dt,
    int max_iter,
    double tol) const
{
    return compute_lqr_gain(model, x_trim, VectorXd::Zero(model.input_dim()), dt, max_iter, tol);
}

// MPCLimits
MPCLimits::MPCLimits(const SystemModel& model) : // default to no constraints
    u_min(VectorXd::Constant(model.input_dim(), -1e10)),
    u_max(VectorXd::Constant(model.input_dim(), 1e10)),
    x_min(VectorXd::Constant(model.state_dim(), -1e10)),
    x_max(VectorXd::Constant(model.state_dim(), 1e10)),
    delta_u_min(VectorXd::Constant(model.input_dim(), -1e10)),
    delta_u_max(VectorXd::Constant(model.input_dim(), 1e10))
{}

// MPCController
MPCController::MPCController(const SystemModel& model, MPCConfig config, const MPCWeights& weights, const MPCLimits& limits) :
    model_(model), 
    config_(config), 
    weights_(weights),
    limits_(limits),
    solve_time_(0.0),
    previous_u_(VectorXd::Zero(model.input_dim())) { }

MatrixXd MPCController::get_predicted_trajectory() const {
    return Eigen::Map<const Eigen::MatrixXd>(predicted_trajectory_.data(),                  // Map the predicted trajectory to a matrix of shape (N, state_dim)
                                             model_.state_dim(), config_.N).transpose();
}

double MPCController::get_solve_time() const {
    return solve_time_;
}

VectorXd MPCController::solve(const VectorXd& x0, const VectorXd& x_ref) {
    int n = model_.state_dim();
    int m = model_.input_dim();
    int N = config_.N;

    // Build MPC matrices
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
            A_ij = A_d * A_ij;  // A^(i-j+1) for next iteration
        }
    }


    // d_prev for delta_u constraints
    VectorXd d_prev = VectorXd::Zero(m*N);
    d_prev.head(m) = previous_u_;

    // Build D for delta_u constraints
    MatrixXd D = MatrixXd::Zero(m*N, m*N);
    for (int k = 0; k < N; k++) {
        D.block(k*m, k*m, m, m) = MatrixXd::Identity(m, m);
        if (k > 0)
            D.block(k*m, (k-1)*m, m, m) = -MatrixXd::Identity(m, m);
    }

    // Build Q_bar and R_bar
    MatrixXd Q_bar = MatrixXd::Zero(n*N, n*N);
    MatrixXd R_bar = MatrixXd::Zero(m*N, m*N);

    for (int i = 0; i < N; i++) {
        Q_bar.block(i*n, i*n, n, n) = weights_.Q;
        R_bar.block(i*m, i*m, m, m) = weights_.R;
    }
    Q_bar.block((N-1)*n, (N-1)*n, n, n) = weights_.Qf; // terminal cost

    // Build S_bar
    MatrixXd S_bar = MatrixXd::Zero(m*N, m*N);
    for (int i = 0; i < N; i++)
        S_bar.block(i*m, i*m, m, m) = weights_.S;


    // Build P and q
    MatrixXd P = Su.transpose() * Q_bar * Su + R_bar + D.transpose() * S_bar * D;
    VectorXd q = Su.transpose() * Q_bar * (Sx * x0 - x_ref) - D.transpose() * S_bar * d_prev; 

    // Convert to sparse
    SparseMatrix<double> P_sparse = P.sparseView();


    // Input, state, and rate constraints
    // upper and lower bounds
    VectorXd lb(m*N + n*N + m*N);
    VectorXd ub(m*N + n*N + m*N);
    lb << limits_.u_min.replicate(N,1), state_lower_bounds(Sx, x0), limits_.delta_u_min.replicate(N,1) + d_prev;
    ub << limits_.u_max.replicate(N,1), state_upper_bounds(Sx, x0), limits_.delta_u_max.replicate(N,1) + d_prev;


    // Constraint matrix: [input bounds; state bounds; rate bounds]
    MatrixXd A_con_dense(m*N + n*N + m*N, m*N);
    A_con_dense << MatrixXd::Identity(m*N, m*N), Su, D;
    SparseMatrix<double> A_con = A_con_dense.sparseView();


    // Solve QP
    auto t_start = std::chrono::high_resolution_clock::now();   // Start timer
    
    // Setup solver
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.data()->setNumberOfVariables(m * N);
    solver.data()->setNumberOfConstraints(m*N + n*N + m*N);
    solver.data()->setLinearConstraintsMatrix(A_con);
    solver.data()->setLowerBound(lb);
    solver.data()->setUpperBound(ub);
    solver.data()->setHessianMatrix(P_sparse);
    solver.data()->setGradient(q);
    solver.initSolver();
    // Run sover
    solver.solveProblem();
    
    auto t_end = std::chrono::high_resolution_clock::now();     // End timer
    solve_time_ = std::chrono::duration<double>(t_end - t_start).count();

    // Output processing
    // Extract first control input
    VectorXd U = solver.getSolution();
    if (!U.allFinite() || U.cwiseAbs().maxCoeff() > 1e3) {
        return previous_u_;   // bad solve (infeasible under disturbance) -> hold last command
    }

    // Predicted trajectory
    predicted_trajectory_ = Sx * x0 + Su * U;
    // Store previous control for delta_u constraints in next iteration
    previous_u_ = U.segment(0, m);
    
    return U.segment(0, m);
}

VectorXd MPCController::state_upper_bounds(const MatrixXd& Sx, const VectorXd& x0) const {
    return limits_.x_max.replicate(config_.N, 1) - Sx * x0;
}

VectorXd MPCController::state_lower_bounds(const MatrixXd& Sx, const VectorXd& x0) const {
    return limits_.x_min.replicate(config_.N, 1) - Sx * x0;
}