#include <cassert>
#include <experimental/filesystem>
#include <iostream>
#include <vector>

#include "pose_estimation.hpp"
#include "triangulation.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

namespace fs = std::experimental::filesystem;
using namespace PoseEstimation;
using namespace cv;
using namespace std;

int main() {
  fs::path res_dir = fs::path(RES_DIR);
  fs::path img1_path = res_dir / "1.png";
  fs::path img2_path = res_dir / "2.png";
  fs::path img1_depth_path = res_dir / "1_depth.png";
  fs::path img2_depth_path = res_dir / "2_depth.png";

  cv::Mat img1 = (cv::imread(img1_path, 0));
  cv::Mat img2 = (cv::imread(img2_path, 0));
  assert(img1.data && img2.data && "can not load images!");

  cv::Mat K =
      (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  std::vector<cv::KeyPoint> kpts1, kpts2;
  std::vector<cv::DMatch> matches;
  spdlog::info("find feature");
  find_feature_matches(img1, img2, kpts1, kpts2, matches);
  spdlog::info("In Total, we get {} set of feature points", matches.size());

  // depth image
  Mat d1 = imread(img1_depth_path, cv::IMREAD_UNCHANGED);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  for (const auto &m : matches){
    ushort d = d1.ptr<unsigned short>(int(kpts1[m.queryIdx].pt.y))[int(kpts1[m.queryIdx].pt.x)];
    if (d == 0) // bad depth
      continue;
    float dd1 = d / 5000.f;
    Point2d p1 = pixel2cam(kpts1[m.queryIdx].pt, K);
    pts_3d.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1)); // obj point
    pts_2d.push_back(kpts2[m.trainIdx].pt); // img point
  }
  
  // Estimate motion between two frames
  cv::Mat R, t;

  // 2d - 2d estimation
  pose_estimation_2d2d(kpts1, kpts2, matches, K, R, t, false);
  cout << "2d-2d : R:\n";
  cout << R << endl << endl;
  // double tmp = t.at<double>(2, 0);
  // t.at<double>(0, 0) /= tmp;
  // t.at<double>(1, 0) /= tmp;
  // t.at<double>(2, 0) /= tmp;

  // Check E = t^R*scale
  // Mat t_x = (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
  //            t.at<double>(2, 0), 0, -t.at<double>(0, 0), -t.at<double>(1, 0),
  //            t.at<double>(0, 0), 0);
  // cout << "t^R=\n" << t_x * R << endl;

  // Check epipolar constraint E = [b]^R
  //   for (const auto &m : matches) {
  //     Point2d pt1 = pixel2cam(kpts1[m.queryIdx].pt, K);
  //     Mat X1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
  //     Point2d pt2 = pixel2cam(kpts2[m.trainIdx].pt, K);
  //     Mat X2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1.0);

  //     Mat d = X2.t() * t_x * R * X1;
  //     cout << "epipolar constraint = \n" << d << endl;
  //   }
  // std::vector<cv::Point3d> points;
  // triangulationOpenCV(kpts1, kpts2, matches, K, R, t, points);
  // for (const auto & p : points)
  //   std::cout << p << std::endl;

  /// 3d - 2d estimation
  // OpenCV solvePnP
  spdlog::info("3d-2d pairs: {}", pts_3d.size());
  spdlog::stopwatch sw;
  solvePnP(pts_3d, pts_2d, K, Mat(), R, t, false);
  Rodrigues(R, R);
  spdlog::info("solve pnp in opencv cost time: {} seconds", sw);
  cout << "opencv solve pnp\n";
  cout << "R:\n" << R << endl;
  cout << "t:\n" << t.t() << endl << endl;

  // Gaussian Newton
  VecVector3d pts_3d_eigen;
  pts_3d_eigen.reserve(pts_3d.size());
  VecVector2d pts_2d_eigen;
  pts_2d_eigen.reserve(pts_3d.size());
  for (size_t i=0; i<pts_3d.size(); ++i)
  {
    pts_3d_eigen.emplace_back(Eigen::Vector3d{pts_3d[i].x, pts_3d[i].y, pts_3d[i].z});
    pts_2d_eigen.emplace_back(Eigen::Vector2d{pts_2d[i].x, pts_2d[i].y});
  }
  spdlog::info("Calling bundle adjustment by gauss newton");
  Sophus::SE3d pose_gn;
  sw.reset();
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
  spdlog::info("solve pnp by gauss newton cost time: {} seconds", sw);
  cout << "pose by Gaussian Newton\n";
  cout << "Rt:\n" << pose_gn.matrix() << endl << endl;

  // g2o
  spdlog::info("Calling bundle adjustment by g2o");
  Sophus::SE3d pose_g2o;
  sw.reset();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  spdlog::info("solve pnp by g2o cost time: {} seconds", sw);
  cout << "pose by g2o\n";
  cout << "Rt:\n" << pose_gn.matrix() << endl << endl;

  /// 3D - 3D
  Mat depth1 = imread(img1_depth_path, IMREAD_UNCHANGED);
  Mat depth2 = imread(img2_depth_path, IMREAD_UNCHANGED);
  vector<cv::Point3f> pts1, pts2;

  for (const auto m : matches)
  {
    ushort d1 = depth1.ptr<unsigned short>(int(kpts1[m.queryIdx].pt.y))[int(kpts1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(kpts2[m.trainIdx].pt.y))[int(kpts2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)
      continue;
    Point2d p1 = pixel2cam(kpts1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(kpts2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.f;
    float dd2 = float(d2) / 5000.f;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }

  // SVD
  spdlog::info("ICP via SVD");
  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "R = \n" << R << endl;
  cout << "t =\n" << t << endl;
  cout << "R_inv = \n" << R.t() << endl;
  cout << "t_inv = \n" << -R.t() * t << endl << endl;

  // g2o
  cv::Mat R_, t_;
  spdlog::info("ICP via G2O");
  bundleAdjustment(pts1, pts2, R_, t_);
  cout << "R = \n" << R_ << endl;
  cout << "t =\n" << t_ << endl;
  cout << "R_inv = \n" << R_.t() << endl;
  cout << "t_inv = \n" << -(R_.t() * t_) << endl << endl;

  return 0;
}