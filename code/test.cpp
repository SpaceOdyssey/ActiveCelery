#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include "celerite/celerite.h"

using Eigen::VectorXd;

int main () {
    // Choose some demo parameters for the solver
    int j_real = 2, j_complex = 1;
    VectorXd alpha_real(j_real),
             beta_real(j_real),
             alpha_complex_real(j_complex),
             alpha_complex_imag(j_complex),
             beta_complex_real(j_complex),
             beta_complex_imag(j_complex);
    alpha_real << 1.0, 0.3;
    beta_real << 0.5, 3.5;
    alpha_complex_real << 1.0;
    alpha_complex_imag << 0.1;
    beta_complex_real << 3.0;
    beta_complex_imag << 1.0;

    // Generate some fake data
    int N = 500;
    srand(42);
    VectorXd x = VectorXd::Random(N),
             yvar = VectorXd::Random(N),
             y;
    yvar.array() *= 0.1;
    yvar.array() += 1.0;
    std::sort(x.data(), x.data() + x.size()); // The independent coordinates must be sorted
    y = sin(x.array());

    // Set up the solver
    celerite::solver::BandSolver<double> solver;
    solver.compute(
        alpha_real, beta_real,
        alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag,
        x, yvar  // Note: this is the measurement _variance_
    );

    std::cout << solver.log_determinant() << std::endl;
    std::cout << solver.dot_solve(y) << std::endl;

    return 0;
}

