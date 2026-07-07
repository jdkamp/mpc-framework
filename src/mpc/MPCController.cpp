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

    int nU = m * N;     // number control variables
    int nS = n * N;     // one slack per state-constraint
    int nVars = nU + nS;// augmented decision vector
    double rho = 1e3;   // L1 slack penalty
    const double INF = 1e30;

    // Build MPC matrices
    // Linearization
    VectorXd u0 = VectorXd::Zero(m);
    MatrixXd A = model_.jacobian_x(x0, u0);
    MatrixXd B = model_.jacobian_u(x0, u0);

    // Discretization using Euler method
    MatrixXd A_d = MatrixXd::Identity(n, n) + config_.dt * A;
    MatrixXd B_d = config_.dt * B;

    // Affine offset of the linearization: x_{k+1} = A_d x_k + B_d u_k + c.
    // Without it the prediction double-counts the free response away from (x0, u0).
    VectorXd c = config_.dt * (model_.dynamics(x0, u0) - A * x0 - B * u0);


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

    // Lifted affine offset: off_k = A_d off_{k-1} + c, and the resulting free response
    VectorXd C_lift(n*N);
    VectorXd off = VectorXd::Zero(n);
    for (int k = 0; k < N; k++) {
        off = A_d * off + c;
        C_lift.segment(k*n, n) = off;
    }
    VectorXd x_free = Sx * x0 + C_lift;   // predicted trajectory for U = 0


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
    VectorXd q = Su.transpose() * Q_bar * (x_free - x_ref) - D.transpose() * S_bar * d_prev;
    
    // Augmented P and q for slack
    MatrixXd P_aug = MatrixXd::Zero(nVars, nVars);
    P_aug.topLeftCorner(nU, nU) = P;
    P_aug.bottomRightCorner(nS, nS) = 1e-6 * MatrixXd::Identity(nS, nS);
    SparseMatrix<double> P_sparse = P_aug.sparseView();

    VectorXd q_aug(nVars);
    q_aug << q, rho * VectorXd::Ones(nS);  // L1 penalty on each slack


    // Input, state, and rate constraints
    // upper and lower bounds
    int nCon = nU + nS + nS + nU + nS;     // input + state-upper + state-lower + rate + slack>=0
    MatrixXd A_con_dense(nCon, nVars);
    A_con_dense.setZero();
    VectorXd lb(nCon), ub(nCon);
    int r = 0;

    // 1) input bounds:  [I | 0]
    A_con_dense.block(r, 0, nU, nU) = MatrixXd::Identity(nU, nU);
    lb.segment(r, nU) = limits_.u_min.replicate(N, 1);
    ub.segment(r, nU) = limits_.u_max.replicate(N, 1);
    r += nU;

    // 2) state upper (soft):  Su*U - s <= x_upper   ->  [Su | -I],  ub = x_upper
    A_con_dense.block(r, 0, nS, nU)  = Su;
    A_con_dense.block(r, nU, nS, nS) = -MatrixXd::Identity(nS, nS);
    lb.segment(r, nS).setConstant(-INF);
    ub.segment(r, nS) = state_upper_bounds(x_free, x0);
    r += nS;

    // 3) state lower (soft):  Su*U + s >= x_lower   ->  [Su | +I],  lb = x_lower
    A_con_dense.block(r, 0, nS, nU)  = Su;
    A_con_dense.block(r, nU, nS, nS) = MatrixXd::Identity(nS, nS);
    lb.segment(r, nS) = state_lower_bounds(x_free, x0);
    ub.segment(r, nS).setConstant(INF);
    r += nS;

    // 4) rate bounds:  [D | 0]
    A_con_dense.block(r, 0, nU, nU) = D;
    lb.segment(r, nU) = limits_.delta_u_min.replicate(N, 1) + d_prev;
    ub.segment(r, nU) = limits_.delta_u_max.replicate(N, 1) + d_prev;
    r += nU;

    // 5) slack >= 0:  [0 | I]
    A_con_dense.block(r, nU, nS, nS) = MatrixXd::Identity(nS, nS);
    lb.segment(r, nS).setConstant(0.0);
    ub.segment(r, nS).setConstant(INF);

    SparseMatrix<double> A_con = A_con_dense.sparseView();



    // Solve QP
    auto t_start = std::chrono::high_resolution_clock::now();   // Start timer
    
    // Setup solver
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.data()->setNumberOfVariables(nVars);
    solver.data()->setNumberOfConstraints(nCon);
    solver.data()->setLinearConstraintsMatrix(A_con);
    solver.data()->setLowerBound(lb);
    solver.data()->setUpperBound(ub);
    solver.data()->setHessianMatrix(P_sparse);
    solver.data()->setGradient(q_aug);
    solver.initSolver();
    // Run sover
    solver.solveProblem();
    
    auto t_end = std::chrono::high_resolution_clock::now();     // End timer
    solve_time_ = std::chrono::duration<double>(t_end - t_start).count();

    // Output processing
    // Extract first control input
    VectorXd Z = solver.getSolution();
    if (!Z.allFinite() || Z.cwiseAbs().maxCoeff() > 1e6) {
        std::cout << "Bad solve \n";
        return previous_u_;   // bad solve (infeasible under disturbance) -> hold last command
    }
    VectorXd U = Z.head(nU);

    // Predicted trajectory
    predicted_trajectory_ = x_free + Su * U;
    // Store previous control for delta_u constraints in next iteration
    previous_u_ = U.segment(0, m);
    
    return U.segment(0, m);
}

VectorXd MPCController::state_upper_bounds(const VectorXd& x_free, const VectorXd& x0) const {
    return limits_.x_max.replicate(config_.N, 1) - x_free;
}

VectorXd MPCController::state_lower_bounds(const VectorXd& x_free, const VectorXd& x0) const {
    return limits_.x_min.replicate(config_.N, 1) - x_free;
}