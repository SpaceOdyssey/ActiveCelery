#ifndef Celery_Data
#define Celery_Data

#include <vector>
#include <Eigen/Dense>

namespace Celery
{

class Data
{
    private:
        std::vector<double> t, y, t_predict;
        double y_mean, y_sd;
        Eigen::VectorXd tt, yy, tt_predict;

    public:
        Data();
        void load(const char* filename);

        // Getters
        const std::vector<double>& get_t() const { return t; }
        const std::vector<double>& get_y() const { return y; }
        const std::vector<double>& get_t_predict() const { return t_predict; }

        // Getters of eigen vectors
        const Eigen::VectorXd& get_tt() const { return tt; }
        const Eigen::VectorXd& get_yy() const { return yy; }
        const Eigen::VectorXd& get_tt_predict() const { return tt_predict; }

        // Getters for y-scale normalisation
        double get_y_mean() const { return y_mean; }
        double get_y_sd() const { return y_sd; }

        // Summaries
        double get_t_range() const { return (t.back() - t[0]); }

    // Singleton
    private:
        static Data instance;
    public:
        static Data& get_instance() { return instance; }
};

} // namespace

#endif
