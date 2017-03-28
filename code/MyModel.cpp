#include "Data.h"
#include "MyModel.h"
#include "DNest4/code/DNest4.h"

namespace Celery
{

MyModel::MyModel()
:modes(3,        // Dimensionality of a component
       10,       // Maximum number of components
       false,    // Fixed number of components?
       MyConditionalPrior(0.0,      // Conditional prior
                          Data::get_instance().get_t_range()),
       DNest4::PriorType::log_uniform)
{

}

void MyModel::from_prior(DNest4::RNG& rng)
{
    modes.from_prior(rng);
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    logH += modes.perturb(rng);

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    // Get the RJObject components
    const auto& components = modes.get_components();
    size_t num_modes = components.size();

    // Only need these four
    Eigen::VectorXd a(num_modes);
    Eigen::VectorXd b(num_modes);
    Eigen::VectorXd c(num_modes);
    Eigen::VectorXd d(num_modes);
    for(size_t i=0; i<components.size(); ++i)
    {
        a(i) = components[i][0];                // Amplitude
        b(i) = 0.0;                             // Always zero for oscillations
        c(i) = 1.0 / (components[i][1]*components[i][2]); // 1 / (mode lifetime)
        d(i) = 2.0 * M_PI / components[i][1];   // Angular frequency
    }

    // Grab the data
    const Data& data = Data::get_instance();

    // Two empty vectors
    Eigen::VectorXd junk1, junk2;

    // Celerite solver
    celerite::solver::BandSolver<double> solver;
    solver.compute(junk1, junk2,
                   a, b, c, d,
                   data.get_tt(), data.get_var());

    logL += -0.5*log(2*M_PI)*data.get_y().size();
    logL += -0.5*solver.log_determinant();
    logL += -0.5*solver.dot_solve(Data::get_instance().get_yy());

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    modes.print(out);
    out<<' ';
}

std::string MyModel::description() const
{
    return std::string("");
}

} // namespace Celery

