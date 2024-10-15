#ifndef Celery_MyConditionalPrior
#define Celery_MyConditionalPrior

#include <boost/math/distributions/normal.hpp>
#include "DNest4/code/DNest4.h"

namespace Celery
{

class MyConditionalPrior:public DNest4::ConditionalPrior
{
    private:
        // Hyperpriors on quality factor.

        // Component priors.
        DNest4::TruncatedCauchy prior_amplitude;
        DNest4::Uniform prior_period;
        DNest4::TruncatedCauchy prior_quality;

        double perturb_hyperparameters(DNest4::RNG& rng);

    public:
        MyConditionalPrior();

        void from_prior(DNest4::RNG& rng);

        double log_pdf(const std::vector<double>& vec) const;
        void from_uniform(std::vector<double>& vec) const;
        void to_uniform(std::vector<double>& vec) const;

        void print(std::ostream& out) const;
};

} // namespace Celery

#endif
