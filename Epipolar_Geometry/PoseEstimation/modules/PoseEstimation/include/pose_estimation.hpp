#ifndef POSE_ESTIMATION__HPP
#define POSE_ESTIMATION_HPP

#include <Eigen/Core>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <vector>

using VecVector2d = std::vector<Eigen::Vector2d>;
using VecVector3d = std::vector<Eigen::Vector3d>;

namespace PoseEstimation {
void pose_estimation_2d2d(const std::vector<cv::KeyPoint> &kpts1,
                          const std::vector<cv::KeyPoint> &kpts2,
                          const std::vector<cv::DMatch> &matches,
                          const cv::Mat &K, cv::Mat &R, cv::Mat &t,
                          bool verbose = true);

void bundleAdjustmentGaussNewton(const VecVector3d &pts_3d,
                                 const VecVector2d &pts_2d, const cv::Mat &K,
                                 Sophus::SE3d &pose);

void bundleAdjustmentG2O(const VecVector3d &pts_3d, const VecVector2d &pts_2d,
                         const cv::Mat &K, Sophus::SE3d &pose);

/// vertex and edges used in g2o BA
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

  // Left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4],
        update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(std::istream &in) override { return true; }
  virtual bool write(std::ostream &out) const override { return true; }
};

class EdgeProjection
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K)
      : _pos3d(pos), _K(K) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2,
        -fx - fx * X * X / Z2, fx * Y / Z, 0, -fy / Z, fy * Y / (Z * Z),
        fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(std::istream &in) override { return true; }
  virtual bool write(std::ostream &out) const override { return true; }

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};

/*
class EdgeProjectXYZRGBDPoseOnly
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  virtual bool read(std::istream &in) override { return true; }
  virtual bool write(std::ostream &out) const override { return true; }

protected:
  Eigen::Vector3d _point;
};

*/
class EdgeProjectXYZRGBDPoseOnly
    : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(std::istream &in) {return true;}

  bool write(std::ostream &out) const {return true;}

protected:
  Eigen::Vector3d _point;
};

void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1,
                          const std::vector<cv::Point3f> &pts2, cv::Mat &R,
                          cv::Mat &t);

void bundleAdjustment(const std::vector<cv::Point3f> &pts1,
                      const std::vector<cv::Point3f> &pts2, cv::Mat &R,
                      cv::Mat &t);
} // namespace PoseEstimation
#endif