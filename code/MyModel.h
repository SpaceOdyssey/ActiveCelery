#ifndef Celery_MyModel
#define Celery_MyModel

#include "DNest4/code/DNest4.h"
#include "MyConditionalPrior.h"
#include <ostream>
#include "celerite/celerite.h"

namespace Celery
{

class MyModel
{
    private:
        // Maximum number of modes
        static constexpr size_t max_num_modes = 20;

        // Circadian component
        double log_amplitude_circ;
        DNest4::Gaussian prior_log_amplitude_circ;

        double period_circ;
        DNest4::Gaussian prior_period_circ;

        double quality_circ;
        DNest4::TruncatedCauchy prior_quality_circ;

        // The modes
        DNest4::RJObject<MyConditionalPrior> modes;

        // Noise model
        double sigma;
        DNest4::TruncatedCauchy prior_sigma;

        // Predictive models.
        Eigen::VectorXd predictive_mean() const;
        Eigen::VectorXd predictive_ultradian_mean() const;
        Eigen::VectorXd predictive_circadian_mean() const;
        Eigen::VectorXd predictive_corr_noise_mean() const;

    public:
        // Constructor only gives size of params
        MyModel();

        // Generate the point from the prior
        void from_prior(DNest4::RNG& rng);

        // Metropolis-Hastings proposals
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        double log_likelihood() const;

        // Print to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;



};

} // namespace Celery

#endif
