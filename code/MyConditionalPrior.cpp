#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include <cmath>
#include <exception>
#include "Data.h"

namespace Celery
{

MyConditionalPrior::MyConditionalPrior()
{
    // Component priors.
    prior_log_amplitude = DNest4::Gaussian(0.0, 2.5);
    prior_period = DNest4::Uniform(1/24.0, 1.0);
    prior_quality = DNest4::TruncatedCauchy(0.0, 100, 1E-6);
}

void MyConditionalPrior::from_prior(DNest4::RNG& rng)
{

}

double MyConditionalPrior::perturb_hyperparameters(DNest4::RNG& rng)
{
    double logH = 0.0;

    return logH;
}

// component = {amplitude, period, quality}
double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    double logP = 0.0;

    if(vec[0] <= 0.0)
        return -1E300;
    if(vec[1] <= 0.0)
        return -1E300;
    if(vec[2] < 0.0)
        return -1E300;

    logP += prior_log_amplitude.log_pdf(log(vec[0]));
    logP += prior_period.log_pdf(vec[1]);
    logP += prior_quality.log_pdf(vec[2]);

    return logP;
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    vec[0] = exp(prior_log_amplitude.cdf_inverse(vec[0]));
    vec[1] = prior_period.cdf_inverse(vec[1]);
    vec[2] = prior_quality.cdf_inverse(vec[2]);
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = prior_log_amplitude.cdf(log(vec[0]));
    vec[1] = prior_period.cdf(vec[1]);
    vec[2] = prior_quality.cdf(vec[2]);
}

void MyConditionalPrior::print(std::ostream& out) const
{

}

} // namespace Celery
