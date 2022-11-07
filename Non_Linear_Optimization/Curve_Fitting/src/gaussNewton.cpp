#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

int main() {
  double ar = 1.0, br = 2.0, cr = 1.0;  // GT
  double ae = 2.0, be = -1.0, ce = 5.0; // initial estimate
  int N = 100;                          // Num of data points
  double w_sigma = 1.0;                 // sigma of noise
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;

  std::vector<double> x_data, y_data; // data
  for (int i = 0; i < N; ++i) {
    double x = static_cast<double>(i) / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) +
                     rng.gaussian(w_sigma * w_sigma));
  }

  // Start Gauss-Newton iterations
  int iterations = 100;
  double cost = 0, last_cost = 0;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  /*
    residual function f : yi - exp(ae * x^2 + be*x + ce)
        - ae : estimate of a
        - be : estimate of b
        - ce : estiamte of c
    they are updated at each iteration
    goal d_abc[3] such that
        ae + d_abc[0]
        be + d_abc[1]
        ce + d_abc[2]
    approach min f
  */
  for (int iter = 0; iter < iterations; ++iter) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    cost = 0;

    for (int i = 0; i < N; ++i) {
      double xi = x_data[i], yi = y_data[i]; // i-th data
      double error = yi - exp(ae * xi * xi + be * xi + ce); // error = f(x_i)
      Eigen::Vector3d J; // jacobian
      double exp_term = exp(ae * xi * xi + be * xi + ce);
      J[0] = -xi * xi * exp_term; // de/da
      J[1] = -xi * exp_term;      // de/db
      J[2] = -exp_term;           // de/dc

      H += inv_sigma * inv_sigma * J * J.transpose();
      g += inv_sigma * inv_sigma * -error * J;

      cost += error * error;
    }

    // Solve Hx = b
    Eigen::Vector3d d_abc = H.ldlt().solve(g);
    if (isnan(d_abc[0])) {
      std::cout << "result is nan!\n";
      break;
    }

    if (iter > 0 && cost >= last_cost) {
      std::cout << "cost: " << cost << ">= last cost: " << last_cost
                << ", break." << std::endl;
      break;
    }

    ae += d_abc[0];
    be += d_abc[1];
    ce += d_abc[2];

    last_cost = cost;

    std::cout << "total cost: " << cost << ", \t\tupdate: " << d_abc.transpose()
              << "\t\testimated params: " << ae << "," << be << "," << ce
              << std::endl;
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "time: " << ts / 1000.0 << std::endl;
  std::cout << "estimated abc = " << ae << ", " << be << ", " << ce << std::endl;

  return 0;
}