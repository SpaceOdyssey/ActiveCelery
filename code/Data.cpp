#include "Data.h"
#include <exception>
#include <fstream>
#include <iostream>

namespace Celery
{

Data Data::instance;

Data::Data()
{

}

void Data::load(const char* filename)
{
    std::fstream fin(filename, std::ios::in);
    if(!fin)
    {
        std::cerr<<"# Error. Couldn't open file "<<filename<<"."<<std::endl;
        return;
    }

    // Empty the vectors
    t.clear();
    y.clear();

    double temp1, temp2;
    while(fin>>temp1 && fin>>temp2)
    {
        t.push_back(temp1);
        y.push_back(temp2);

        if(t.size() >= 2 && t.back() < t[t.size() - 2])
            throw std::invalid_argument("Unsorted t-values in file.");
    }
    std::cout<<"# Loaded "<<t.size()<<" data points from file "
            <<filename<<"."<<std::endl;
    fin.close();

    // Normalise y-scale. This allows the priors for amplitudes to be on a
    // unitless scale.
    double y_sum = 0.0;
    for (size_t i=0; i<y.size(); ++i) {
        y_sum += y[i];
    }
    y_mean = y_sum/y.size();

    double y_sumsq = 0.0;
    for (size_t i=0; i<y.size(); ++i) {
        y_sumsq += pow(y[i] - y_mean, 2);
    }
    y_sd = std::sqrt(y_sumsq/(y.size() - 1.0));

    for (size_t i=0; i<y.size(); ++i) {
        y[i] = (y[i] - y_mean)/y_sd;
    }

    // Copy into eigen vectors
    tt.resize(t.size());
    yy.resize(y.size());
    for(size_t i=0; i<y.size(); ++i)
    {
        tt(i) = t[i];
        yy(i) = y[i];
    }

    // Posterior prediction locations.
    double dt = 1.0/(6.0*24.0);  // Assuming 10 minute bins.
    size_t n_t = std::ceil((t.back() - t[0])/dt);
    for (size_t i = 0; i<=n_t; ++i) {
        t_predict.push_back(t[0] + i*dt);
    }

    // Copy into eigen vectors
    tt_predict.resize(t_predict.size());
    for(size_t i=0; i<t_predict.size(); ++i) {
        tt_predict(i) = t_predict[i];
    }

    // Write predictive t to file:
    std::ofstream outFile("t_predict.txt");
    for(size_t i=0; i<t_predict.size(); ++i) {
        outFile << t_predict[i] << std::endl;
    }
    outFile.close();

}

} // namespace Celery
