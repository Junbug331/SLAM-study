#include <Eigen/Dense>
#include <future>
#include <iostream>
#include <thread>

#include "triangulation.hpp"
#include "utils.hpp"
/*
Cross Product
i  j  k
a1 a2 a3
b1 b2 b3

a x b =
a2b3 - a3b2
a3b1 - a1b3
a1b2 - a2b1

x     | p1   p2   p3   p4  | X
y = a | p5   p6   p7   p8  | Y
z     | p9   p10  p11  p12 | Z
                             1
P1^T = {p1 p2 p3 p4}
P2^T = {p5 p6 p7 p8}
P3^T = {p9 p10 p11 p12}

X = {X; Y; Z; 1}

x     | P1^T*X |
y = a | P2^T*X |
z     | P3^T*X |

x    | P1^T*X |   | yP3^T*X - zP2^T*X |
y x  | P2^T*X | = | P1^T*X - xP3^T*X  |
1    | P3^T*X |   | xP2^T*X - yP1^T*X |


*/
namespace PoseEstimation {

void triangulationSVD(const std::vector<cv::KeyPoint> &kpts1,
                      const std::vector<cv::KeyPoint> &kpts2,
                      const std::vector<cv::DMatch> &matches, const cv::Mat &K,
                      const cv::Mat &R, const cv::Mat &t,
                      std::vector<cv::Point3d> &points) {

  std::array<Eigen::Vector4d, 3> P, Q;
  P[0] << 1, 0, 0, 0;
  P[1] << 0, 1, 0, 0;
  P[2] << 0, 0, 1, 0;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Q[i](j, 0) = R.at<double>(i, j);
    }
  }

  for (int i = 0; i < 3; ++i)
    Q[i](3, 0) = t.at<double>(i, 0);

  std::vector<Eigen::Vector3d> pts1, pts2;
  pts1.reserve(matches.size());
  pts2.reserve(matches.size());
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  // result point (3D)
  points.clear();
  points.resize(matches.size());

  // std::vector<std::future> futures;
  auto svd_tri = [&](int idx) {
    double x1, y1, x2, y2;
    x1 = (kpts1[matches[idx].queryIdx].pt.x - cx) / fx;
    y1 = (kpts1[matches[idx].queryIdx].pt.y - cy) / fy;
    x2 = (kpts2[matches[idx].trainIdx].pt.x - cx) / fx;
    y2 = (kpts2[matches[idx].trainIdx].pt.y - cy) / fy;

    Eigen::Matrix4d Ai;
    Ai.block<1, 4>(0, 0) = y1 * P[2] - P[1];
    Ai.block<1, 4>(1, 0) = P[0] - x1 * P[2];
    Ai.block<1, 4>(2, 0) = y2 * Q[2] - Q[1];
    Ai.block<1, 4>(3, 0) = Q[0] - x2 * Q[2];

    Ai = Ai.transpose() * Ai;
    Eigen::JacobiSVD<Eigen::Matrix4d, Eigen::ComputeThinU | Eigen::ComputeThinV>
        svd(Ai);
    Eigen::Vector4d v = svd.matrixV().col(3);
    double tmp = v(3, 0);
    double x = v(0, 0) / tmp;
    double y = v(1, 0) / tmp;
    double z = v(2, 0) / tmp;

    points[idx] = {x, y, z};
  };

  // futures for asynchronous programming
  std::vector<std::future<void>> futures;
  futures.reserve(matches.size());
  for (int i = 0; i < matches.size(); ++i) {
    futures.push_back(std::async(svd_tri, i));
  }

  for (int i = 0; i < matches.size(); ++i)
    futures[i].get();
}

void triangulationOpenCV(const std::vector<cv::KeyPoint> &kpts1,
                         const std::vector<cv::KeyPoint> &kpts2,
                         const std::vector<cv::DMatch> &matches,
                         const cv::Mat &K, const cv::Mat &R, const cv::Mat &t,
                         std::vector<cv::Point3d> &points) {
  points.clear();
  points.reserve(matches.size());
  cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
                R.at<double>(0, 2), t.at<double>(0, 0), R.at<double>(1, 0),
                R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
                t.at<double>(2, 0));
  std::vector<cv::Point2f> pts1, pts2;
  pts1.reserve(matches.size());
  pts2.reserve(matches.size());
  for (const auto &m : matches) {
    pts1.emplace_back(pixel2cam(kpts1[m.queryIdx].pt, K));
    pts2.emplace_back(pixel2cam(kpts2[m.trainIdx].pt, K));
  }

  cv::Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts1, pts2, pts_4d);

  // Convert to non-homogeneous coordinates
  for (int i = 0; i < pts_4d.cols; ++i) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0);
    points.emplace_back(
        cv::Point3d(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0)));
  }
}

} // namespace PoseEstimation