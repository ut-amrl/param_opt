#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include <eigen3/Eigen/Dense>
#include <vector>

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


DEFINE_string(dataset_name, "low_speed_optimal_line/curv_greater.csv", "filename to dataset");
DEFINE_int32(n_samples, 41, "he");
DEFINE_int32(n_iters, 10, "iters of ceres solver");

std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> dataset() {
    return {};
}

struct RolloutCostResidual {
    const Eigen::VectorXd costs_;
    const Eigen::MatrixXd state_mat_;

// public:
    RolloutCostResidual(const Eigen::VectorXd &costs, const Eigen::MatrixXd &state_mat) : costs_(costs), state_mat_(state_mat) {}

    template <typename T>
    bool operator()(const T *const w0, const T *const w1, const T *const w2, const T *const w3, T* residual) const {
        Eigen::Matrix<T, 4, 1> weights(*w0, *w1, *w2, *w3);
        residual[0] = (state_mat_ * weights - costs_).norm();
        return true;
    }
};

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // auto dataset = dataset;
    int n_rows = 5;

    ceres::Problem problem_;
    ceres::Solver::Options options_;
    ceres::Solver::Summary summary_;

    options_.linear_solver_type = ceres::DENSE_QR;
    options_.max_num_iterations = FLAGS_n_iters;
    options_.minimizer_progress_to_stdout = false;

    double w0;
    double w1;
    double w2;
    double w3;

    // for each row in dataset, we have an eigen matrix of features and actions, and a vector of costs (3, 2, 1, 0, 1, 2, 3, 4, ...)

    // we have 4 variables we want to optimize over, a vector4f that we multiply with the matrix and subtract from the label, take the mag
    auto data = dataset();

    for (int i  = 0; i < n_rows; i++) {
        Eigen::VectorXd labels = data[i].second;
        Eigen::MatrixXd state_mat = data[i].first;

        CostFunction *ct = new AutoDiffCostFunction<RolloutCostResidual, 1, 1, 1, 1, 1>(
            new RolloutCostResidual(labels, state_mat)
        );
        problem_.AddResidualBlock(ct, new CauchyLoss(0.5), &w0, &w1, &w2, &w3);
    }

    Solve(options_, &problem_, &summary_);

    std::cout << summary_.FullReport() << std::endl;
    std::cout << "weights: " << w0 << " " << w1 << " " << w2 << " " << w3 << std::endl;
    return 0;
}