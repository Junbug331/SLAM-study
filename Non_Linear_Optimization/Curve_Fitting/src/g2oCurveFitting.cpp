#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <iostream>
#include <opencv2/core/core.hpp>

// Function : y = exp(a*x^2 + b*x + c) + w
// Residual f = y - exp(ae*x^2 + be*x + ce) -> error function

// vertex : 3d vector(a, b, c)
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // override the reset function
  virtual void setToOriginImpl() override { _estimate << 0, 0, 0; }

  // Override the plus operator, just plain vector addition
  virtual void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector3d(update);
  }

  // the dummy read and write function
  virtual bool read(std::istream &in) {}
  virtual bool write(std::ostream &out) const {}
};

// edge: 1D error term
class CurveFittingEdge
    : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

  // define the error term computation
  virtual void computeError() override {
    const CurveFittingVertex *v =
        static_cast<const CurveFittingVertex *>(_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    _error(0, 0) = _measurement -
                   std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
  }

  // the jacobian
  virtual void linearizeOplus() override {
    const CurveFittingVertex *v =
        static_cast<const CurveFittingVertex *>(_vertices[0]);

    const Eigen::Vector3d abc = v->estimate();
    double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    _jacobianOplusXi[0] = -_x * _x * y;
    _jacobianOplusXi[1] = -_x * y;
    _jacobianOplusXi[2] = -y;
  }
  virtual bool read(std::istream &in) {}
  virtual bool write(std::ostream &out) const {}

public:
  double _x; // x data, note y is given in _measurement
};

int main() {
  double ar = 1.0, br = 2.0, cr = 1.0;  // GT
  double ae = 2.0, be = -1.0, ce = 5.0; // Initial estimates
  int N = 100;                          // Num of Points
  double w_sigma = 1.0;                 // std of gaussian noise
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;

  std::vector<double> x_data, y_data;
  x_data.reserve(N);
  y_data.reserve(N);
  for (int i = 0; i < N; ++i) {
    double x = static_cast<double>(i) / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) +
                     rng.gaussian(w_sigma * w_sigma));
  }

  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
  using LinearSolverType =
      g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

  // Choose the optimization method from GN, LM, DogLeg
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

  g2o::SparseOptimizer optimizer; // graph optimizer
  optimizer.setAlgorithm(solver); // set the algorithm
  optimizer.setVerbose(true);     // print the results

  // add vertex
  CurveFittingVertex *v = new CurveFittingVertex();
  v->setEstimate(Eigen::Vector3d(ae, be, ce));
  v->setId(0);
  optimizer.addVertex(v);

  // add edges
  for (int i = 0; i < N; ++i) {
    CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    edge->setVertex(0, v);
    edge->setMeasurement(y_data[i]);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 /
                         (w_sigma * w_sigma)); // Covariance matrix
    optimizer.addEdge(edge);
  }

  std::cout << "start optimization" << std::endl;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "time: " << ts / 1000.0 << std::endl;

  Eigen::Vector3d abc_estimate = v->estimate();
  std::cout << "estimated model: " << abc_estimate.transpose() << std::endl;

  return 0;
}