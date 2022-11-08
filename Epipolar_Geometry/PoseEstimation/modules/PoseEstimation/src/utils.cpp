#include "utils.hpp"
#include <spdlog/spdlog.h>

namespace PoseEstimation {

void find_feature_matches(const cv::Mat &img1, const cv::Mat &img2,
                          std::vector<cv::KeyPoint> &kpts1,
                          std::vector<cv::KeyPoint> &kpts2,
                          std::vector<cv::DMatch> &matches) {
  cv::Mat desc1, desc2;

  // Feature Extraction
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  orb->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
  orb->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

  // Feature Matching
  std::vector<std::vector<cv::DMatch>> knn_matches;
  auto matcher =
      cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
  matcher.knnMatch(desc1, desc2, knn_matches, 2);

  // Filter matches using Lowe's ratio
  const float ratio_thresh = 0.7;
  matches.reserve(knn_matches.size());
  for (const auto m : knn_matches) {
    if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance)
      matches.push_back(m[0]);
  }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return {(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
          (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)};
}

} // namespace PoseEstimation