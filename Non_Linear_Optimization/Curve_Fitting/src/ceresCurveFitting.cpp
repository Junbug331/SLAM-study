#include <array>
#include <ceres/ceres.h>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <random>
#include <vector>

// residual functor
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  template <typename T> bool operator()(const T *const abc, T *residual) const {
    // y - exp(ax^2 + bx + c)
    // output dimension is 1 <-- residual[0]
    residual[0] =
        T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
    // residual[0] = static_cast<T>(_y) - ceres::exp(abc[0] *
    // pow(static_cast<T>(_x), 2) + abc[1] * static_cast<T>(_x) + abc[2]);
    return true;
  }

  // x, y data
  const double _x, _y;
};

int main() {
  double ar = 1.0, br = 2.0, cr = 1.0;  // GT
  double ae = 2.0, be = -1.0, ce = 5.0; // initial estimation
  std::size_t N = 100;                  // num of data points
  double w_sigma = 1.0;                 // std of noise
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;

  // Gaussian Noise
  std::random_device dev;
  std::mt19937_64 eng(dev());
  std::normal_distribution<double> dist(0, w_sigma);

  // data
  std::vector<double> x_data, y_data;
  x_data.reserve(N);
  y_data.reserve(N);
  for (int i = 0; i < N; ++i) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) +
                     rng.gaussian(w_sigma * w_sigma));
  }

  double abc[3] = {ae, be, ce};

  // construct the problem in ceres
  ceres::Problem problem;

  for (int i = 0; i < N; ++i) {
    // add i-th residual into the problem
    problem.AddResidualBlock(
        // use auto-diff,
        // template params: residual type, output dimension, input dimension
        new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
            new CURVE_FITTING_COST(x_data[i], y_data[i])),
        nullptr, // kernel function
        abc      // estimated variables
    );
  }

  // set the solver options
  ceres::Solver::Options options;
  options.linear_solver_type =
      ceres::DENSE_NORMAL_CHOLESKY; // use cholesky to solve the normal
                                    // equation
  options.minimizer_progress_to_stdout = true; // verbose true

  ceres::Solver::Summary summary;
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  ceres::Solve(options, &problem, &summary);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "solve time cost: " << ts / 1000.0 << std::endl;

  // get the outputs
  std::cout << summary.BriefReport() << std::endl;
  std::cout << "estimated a, b, c: ";
  for (auto e : abc)
    std::cout << e << ' ';
  std::cout << std::endl;

  return 0;
}