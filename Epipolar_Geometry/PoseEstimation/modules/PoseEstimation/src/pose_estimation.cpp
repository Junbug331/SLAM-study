#include "pose_estimation.hpp"
#include "utils.hpp"
#include <iostream>
#include <spdlog/spdlog.h>

using namespace std;

namespace PoseEstimation {

void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &kpts1,
                          const std::vector<cv::KeyPoint> &kpts2,
                          const std::vector<cv::DMatch> &matches,
                          const cv::Mat &K, cv::Mat &R, cv::Mat &t,
                          bool verbose) {

  // Convert the matching point to the form of vector <Point2f>
  std::vector<cv::Point2f> pts1, pts2;
  pts1.reserve(matches.size());
  pts2.reserve(matches.size());
  for (const auto &m : matches) {
    pts1.push_back(kpts1[m.queryIdx].pt);
    pts2.push_back(kpts2[m.trainIdx].pt);
  }

  // Calculate fundamental matrix using 8 point algorithm
  cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT);
  if (verbose)
    std::cout << "F:\n" << F << std::endl;

  // Calculate Essential Matrix
  cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC);
  if (verbose)
    std::cout << "E:\n" << E << std::endl;

  // Calculate Homography
  // Since the since is not planar, and calculating the homography matrix here
  // is of little significance.
  cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3);
  if (verbose)
    std::cout << "H:\n" << H << std::endl;

  // Recover rotation and translation from the essential matrix
  cv::recoverPose(E, pts1, pts2, R, t);
  cv::normalize(t, t, 1., 0., cv::NORM_L2);
  if (verbose) {
    std::cout << "R:\n" << R << std::endl;
    std::cout << "t:\n" << t << std::endl;
  }
}

void bundleAdjustmentGaussNewton(const VecVector3d &pts_3d,
                                 const VecVector2d &pts_2d, const cv::Mat &K,
                                 Sophus::SE3d &pose) {
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  const int iterations = 10;
  double cost = 0, last_cost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; ++iter) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // Compute cost
    for (int i = 0; i < pts_3d.size(); ++i) {
      Eigen::Vector3d pc = pose * pts_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj((pc[0] * fx / pc[2]) + cx,
                           (pc[1] * fy / pc[2]) + cy);
      // e  = u - 1/s*KTP
      Eigen::Vector2d e = pts_2d[i] - proj;

      cost += e.squaredNorm();

      Eigen::Matrix<double, 2, 6> J;
      J << -fx * inv_z, 0, fx * pc[0] * inv_z2, fx * pc[0] * pc[1] * inv_z2,
          -fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z, 0, -fy * inv_z,
          fy * pc[1] * inv_z2, fy + fy * pc[1] * pc[1] * inv_z2,
          -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;

      H += J.transpose() * J;
      b += -J.transpose() * e;
    }

    Vector6d dx;
    dx = H.ldlt().solve(b);

    if (isnan(dx[0])) {
      spdlog::info("result is nan!");
      break;
    }

    if (iter > 0 && cost >= last_cost) {
      spdlog::info("cost: {}, last cost: {}", cost, last_cost);
      break;
    }

    // Update estimation
    pose = Sophus::SE3d::exp(dx) * pose;
    last_cost = cost;
    spdlog::info("iteration {} cost = {}", iter, cost);
    if (dx.norm() < 1e-6) {
      // converge
      break;
    }
  }
}

void bundleAdjustmentG2O(const VecVector3d &pts_3d, const VecVector2d &pts_2d,
                         const cv::Mat &K, Sophus::SE3d &pose) {
  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>; // pose is 6, landmark is 3
  using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
  );
  g2o::SparseOptimizer optimizer; // graph model
  optimizer.setAlgorithm(solver); // Set up the solver
  optimizer.setVerbose(true); // Turn on Verbose output for debugging

  // vertex
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  // K
  Eigen::Matrix3d K_eigen;
  K_eigen <<
  K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
  K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
  K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges
  int idx = 1;
  for (size_t i= 0; i< pts_2d.size(); ++i)
  {
    auto p2d = pts_2d[i];
    auto p3d = pts_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(idx++);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
  }

  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  pose = vertex_pose->estimate();
}

} // namespace PoseEstimation