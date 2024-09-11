#include "Data.h"
#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>

namespace Celery {

MyModel::MyModel()
:modes(3,                               // Dimensionality of a component
       max_num_modes,                   // Maximum number of components
       false,                           // Fixed number of components?
       MyConditionalPrior(),            // Conditional prior
       DNest4::PriorType::log_uniform)  // Prior on N
{
    // Circadian (~24 hour) rhythm.
    prior_log_amplitude_circ = DNest4::Gaussian(0.0, 2.5);
    prior_period_circ = DNest4::Gaussian(1.0, 3.0/24.0);
    prior_quality_circ = DNest4::TruncatedCauchy(0.0, 14, 1.0);  // Enforcing a high quality oscillator.

    // Noise model.
    prior_sigma = DNest4::TruncatedCauchy(0.0, 1.0, 1E-6);
}

void MyModel::from_prior(DNest4::RNG& rng) {
    // Circadian (~24 hour) rhythm.
    log_amplitude_circ = prior_log_amplitude_circ.generate(rng);
    period_circ = prior_period_circ.generate(rng);
    quality_circ = prior_quality_circ.generate(rng);

    // Components.
    modes.from_prior(rng);

    // Noise model
    sigma = prior_sigma.generate(rng);
}

double MyModel::perturb(DNest4::RNG& rng) {
    double logH = 0.0;
    double rnd = rng.rand();

    if (rnd < 0.8) {
        logH += modes.perturb(rng);
    } else {
        int which = rng.rand_int(4);
        switch (which) {
            case 0:
                logH += prior_log_amplitude_circ.perturb(log_amplitude_circ, rng);
                break;
            case 1:
                logH += prior_period_circ.perturb(period_circ, rng);
                break;
            case 2:
                logH += prior_quality_circ.perturb(quality_circ, rng);
                break;
            case 3:
                logH += prior_sigma.perturb(sigma, rng);
                break;
        }
    }

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    // Get the RJObject components
    const auto& components = modes.get_components();
    size_t num_modes = components.size();

    // Count number of modes with Q < 0.5,
    // as these need *two* Celerite terms.
    size_t lowQ = 0;
    for(size_t i=0; i<components.size(); ++i)
        if(components[i][2] <= 0.5)
            ++lowQ;

    // Only need these four
    Eigen::VectorXd a(1 + num_modes + lowQ);
    Eigen::VectorXd b(1 + num_modes + lowQ);
    Eigen::VectorXd c(1 + num_modes + lowQ);
    Eigen::VectorXd d(1 + num_modes + lowQ);

    double omega0, Q, Qterm, A2;

    // Circadian term.
    A2 = pow(exp(log_amplitude_circ), 2);
    omega0 = 2.0*M_PI/period_circ;
    Q = quality_circ;

    Qterm = sqrt(4*Q*Q - 1.0);
    a(0) = A2;
    b(0) = A2/Qterm;
    c(0) = omega0 / (2*Q);
    d(0) = c(0) * Qterm;

    // Additional terms.
    size_t j=1;
    for(size_t i=0; i<components.size(); ++i) {
        A2 = pow(components[i][0], 2);
        omega0 = 2.0*M_PI/components[i][1];
        Q = components[i][2];

        if(Q >= 0.5) {
            Qterm = sqrt(4*Q*Q - 1.0);
            a(j) = A2;
            b(j) = A2/Qterm;
            c(j) = omega0 / (2*Q);
            d(j) = c(j) * Qterm;
            ++j;
        } else {
            Qterm = sqrt(1.0 - 4*Q*Q);
            a(j)   = 0.5*A2*(1.0 + 1.0/Qterm);
            a(j+1) = 0.5*A2*(1.0 - 1.0/Qterm);
            b(j) = 0.0;
            b(j+1) = 0.0;
            c(j)   = omega0/(2*Q)*(1.0 - Qterm);
            c(j+1) = omega0/(2*Q)*(1.0 + Qterm);
            d(j)   = 0.0;
            d(j+1) = 0.0;
            j += 2;
        }
    }

    // Grab the data
    const Data& data = Data::get_instance();

    // When the imaginary components of the celerite terms are zero, we get an
    // Ornstein-Uhlenbeck component. Which is what is being done here to model
    // longer term correlated noise. Brendon has set alpha_real = 0, which
    // corresponds to the amplitude of this term, and thus is effectively being
    // removed. We may want to reintroduce this later.
    Eigen::VectorXd alpha_real(1), beta_real(1);
    alpha_real(0) = 0.0;
    beta_real(0)  = 1.0;

    // Celerite solver
    celerite::solver::CholeskySolver<double> solver;

    try {
        Eigen::VectorXd var = Eigen::VectorXd::Constant(data.get_y().size(), sigma*sigma);
        solver.compute(0.0,
                       alpha_real, beta_real,
                       a, b, c, d,
                       data.get_tt(), var);
    } catch(...) {
        return -1E300;
    }

    logL += -0.5*log(2*M_PI)*data.get_y().size();
    logL += -0.5*solver.log_determinant();
    logL += -0.5*solver.dot_solve(Data::get_instance().get_yy());

    return logL;
}

void MyModel::print(std::ostream& out) const {
    out << std::setprecision(12);
    modes.print(out);

    // Circadian component
    out << exp(log_amplitude_circ) << " ";
    out << period_circ << " ";
    out << quality_circ << " ";

    // Noise model
    out << sigma;
}

std::string MyModel::description() const {
    std::stringstream s;

    s << "num_dimensions, max_num_components, ";
    s << "num_components, ";

    for(size_t i=0; i<max_num_modes; ++i)
        s << "amplitude[" << i << "], ";
    for(size_t i=0; i<max_num_modes; ++i)
        s << "period[" << i << "], ";
    for(size_t i=0; i<max_num_modes; ++i)
        s << "quality[" << i << "], ";

    // Circadian component
    s << "amplitude[circ], ";
    s << "period[circ], ";
    s << "quality[circ], ";

    // Noise model
    s << "sigma";

    return s.str();
}

} // namespace Celery
