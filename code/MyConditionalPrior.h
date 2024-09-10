#ifndef Celery_MyConditionalPrior
#define Celery_MyConditionalPrior

#include <boost/math/distributions/normal.hpp>
#include "DNest4/code/DNest4.h"

namespace Celery
{

class MyConditionalPrior:public DNest4::ConditionalPrior
{
    private:
        // // Hyperpriors on quality factor.
        // DNest4::Uniform prior_mu_log_quality;
        // DNest4::Uniform prior_sig_log_quality;

        // Component priors.
        DNest4::Gaussian prior_log_amplitude;
        DNest4::Uniform prior_period;
        DNest4::TruncatedCauchy prior_quality;

        // // Hyperparameters
        // double mu_log_quality;
        // double sig_log_quality;

        double perturb_hyperparameters(DNest4::RNG& rng);

    public:
        MyConditionalPrior();

        void from_prior(DNest4::RNG& rng);

        double log_pdf(const std::vector<double>& vec) const;
        void from_uniform(std::vector<double>& vec) const;
        void to_uniform(std::vector<double>& vec) const;

        // A getter for one of the hyperparameters
        // double get_scale_amplitude() const { return scale_amplitude; }

        void print(std::ostream& out) const;
};

} // namespace Celery

#endif
