#include "models/GPResidualModel.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using json = nlohmann::json;

GPResidualModel::GPResidualModel(const std::string& params_path, double wheelbase) :
    bicycle_model_(wheelbase)
{
    // Open and parse JSON file
    std::ifstream f(params_path);
    json params = json::parse(f);

    // Kerne hyperparameters
    sigma_f_sq_ = params["kernel"]["output_scale"];
    l_v_        = params["kernel"]["length_scale"][0];
    l_delta_    = params["kernel"]["length_scale"][1];
    double noise_variance = params["kernel"]["noise_variance"];

    // Training inputs X_train_ and targets y
    auto X      = params["X_train"];
    auto y_json = params["y_train"];
    int N = X.size();

    X_train_.resize(N, 2);
    VectorXd y(N);
    for(int i = 0; i < N; i++) {
        X_train_(i, 0) = X[i][0];
        X_train_(i, 1) = X[i][1];
        y(i)           = y_json[i];
    }

    // Build K, K_inv_, alpha_
    MatrixXd K(N, N);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            K(i, j) = kernel(X_train_.row(i).transpose(), X_train_.row(j).transpose());
        }
    }
    K += noise_variance * MatrixXd::Identity(N, N); // Sklearn's alpha on the diagonal

    K_inv_ = K.inverse();
    alpha_ = K_inv_ * y;
}

// Delegations to the nominal bicycle model

int GPResidualModel::state_dim() const { 
    return bicycle_model_.state_dim(); 
}

int GPResidualModel::input_dim() const {
    return bicycle_model_.input_dim();
}

MatrixXd GPResidualModel::jacobian_x(const VectorXd&x, const VectorXd&u) const {
    MatrixXd J = bicycle_model_.jacobian_x(x, u);
    J(2, 3) += gp_mean_dv(x(3), u(0));
    return J;
}

MatrixXd GPResidualModel::jacobian_u(const VectorXd&x, const VectorXd&u) const {
    MatrixXd J = bicycle_model_.jacobian_u(x, u);
    J(2, 0) += gp_mean_ddelta(x(3), u(0));
    return J;
}


// GP-augmented methods

VectorXd GPResidualModel::dynamics(const VectorXd&x, const VectorXd&u) const {
    VectorXd dx = bicycle_model_.dynamics(x, u);
    dx(2) += gp_mean(x(3), u(0));   // residual on heading rate (2) - featuers v=x(3) and delta=u(0)
    return dx;
}

double GPResidualModel::gp_mean(double v, double delta) const {
    int N = X_train_.rows();
    VectorXd query(2); 
    query << v, delta;

    VectorXd k_star(N);
    for(int i = 0; i < N; i++) {
        k_star(i) = kernel(query, X_train_.row(i).transpose());
    }

    return k_star.dot(alpha_);
}

double GPResidualModel::gp_variance(double v, double delta) const {
    int N = X_train_.rows();
    VectorXd query(2);
    query << v, delta;

    VectorXd k_star(N);
    for(int i = 0; i < N; i++) {
        k_star(i) = kernel(query, X_train_.row(i).transpose());
    }

    return sigma_f_sq_ - k_star.dot(K_inv_ * k_star);
}

VectorXd GPResidualModel::variance(const VectorXd& x, const VectorXd& u) const {
    VectorXd var = VectorXd::Zero(state_dim());
    var(2) = gp_variance(x(3), u(0));
    return var;
}

double GPResidualModel::kernel(const VectorXd&a, const VectorXd& b) const {
    double dv = a(0) - b(0);                            // Difference in velocity
    double dd = a(1) - b(1);                            // Difference in steering angle
    double term_v = (dv / l_v_) * (dv / l_v_);          // Squared distance in velocity normalized by length scale
    double term_d = (dd / l_delta_) * (dd / l_delta_);  // Squared distance in steering angle normalized by length scale
    double exponent = -0.5 * (term_v + term_d);         // RBF kernel exponent
    return sigma_f_sq_ * std::exp(exponent);            // RBF kernel value
}

double GPResidualModel::gp_mean_dv(double v, double delta) const {
    int N = X_train_.rows();
    VectorXd query(2);
    query << v, delta;

    double sum = 0.0;
    for(int i = 0; i < N; i++) {
        double k_i = kernel(query, X_train_.row(i).transpose());
        double v_i = X_train_(i, 0);
        sum += alpha_(i) * k_i * ( -(v - v_i) / (l_v_ * l_v_) );
    }
    return sum;
}

double GPResidualModel::gp_mean_ddelta(double v, double delta) const {
    int N = X_train_.rows();
    VectorXd query(2);
    query << v, delta;

    double sum = 0.0;
    for(int i = 0; i < N; i++) {
        double k_i = kernel(query, X_train_.row(i).transpose());
        double delta_i = X_train_(i, 1);
        sum += alpha_(i) * k_i * ( -(delta - delta_i) / (l_delta_ * l_delta_) );
    }
    return sum;
}
