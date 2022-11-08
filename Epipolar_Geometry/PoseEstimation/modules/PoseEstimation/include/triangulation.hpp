#ifndef TRIANGULATION_HPP
#define TRIANGULATION_HPP

#include <opencv2/opencv.hpp>
#include <vector>


namespace PoseEstimation {

void triangulationSVD(const std::vector<cv::KeyPoint> &kpts1,
                      const std::vector<cv::KeyPoint> &kpts2,
                      const std::vector<cv::DMatch> &matches, const cv::Mat &K,
                      const cv::Mat &R, const cv::Mat &t,
                      std::vector<cv::Point3d> &points);

void triangulationOpenCV(const std::vector<cv::KeyPoint> &kpts1,
                         const std::vector<cv::KeyPoint> &kpts2,
                         const std::vector<cv::DMatch> &matches,
                         const cv::Mat &K, const cv::Mat &R, const cv::Mat &t,
                         std::vector<cv::Point3d> &points);

} // namespace PoseEstimation

#endif