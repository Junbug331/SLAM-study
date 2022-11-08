#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace PoseEstimation {
void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &kpts1,
                          std::vector<cv::KeyPoint> &kpts2,
                          std::vector<cv::DMatch> &matches);


/**
 * @brief Apply inverse intrinsic matrix(pixel coordinate -> camera coordinate)
 * 
 * @param p pixel coordinate in floating point
 * @param K intrinsic matrix 
 * @return cv::Point2d point in camera coordinate 
 */
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);



} // namespace PoseEstimation

#endif