#pragma once
#ifndef DENSE_MAPPING_HPP
#define DENSE_MAPPING_HPP

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class DenseMono
{
private:
    /// parameters
    int boarder = 20;        // image borader
    int width = 640;         // image width
    int height = 480;        // image height
    int ncc_window_size = 3; // half window size of NCC
    double min_cov = 0.1;    // converge criteria: minimal cov
    double max_cov = 10;     // disconverge criteria: maximal cov

    double fx = 481.2f; // camera intrinsics
    double fy = -480.0f;
    double cx = 319.5f;
    double cy = 239.5f;
    int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // area of NCC
public:
    DenseMono() {}
    DenseMono(int boarder, int width, int height, int ncc_window_size, double min_cov, double max_cov, const cv::Mat &K)
        : boarder(boarder), width(width), height(height), ncc_window_size(ncc_window_size), min_cov(min_cov), max_cov(max_cov)
    {
        fx = K.at<double>(0, 0);
        fy = K.at<double>(1, 1);
        cx = K.at<double>(0, 2);
        cy = K.at<double>(1, 2);
        ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // area of NCC
    }
    DenseMono(int boarder, int width, int height, int ncc_window_size, double min_cov, double max_cov, double fx, double fy, double cx, double cy)
        : boarder(boarder), width(width), height(height), ncc_window_size(ncc_window_size), min_cov(min_cov), max_cov(max_cov),
          fx(fx), fy(fy), cx(cx), cy(cy), ncc_area((2 * ncc_window_size + 1) * (2 * ncc_window_size + 1)) {}

    /**
     * @brief update depth using new images
     * In the update function, we traverse each pixel of the reference frame, first look
     * for an epipolar match in the current frame. If it can match, use the epipolar
     * match to update the estimation of the depth map.
     *
     * @param ref            reference image
     * @param curr           current image
     * @param T_C_R          matrix from ref to cur
     * @param depth          depth estimation
     * @param depth_cov2     covariance of depth
     * @return true          success
     * @return false         fail
     */
    bool update(
        const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R,
        cv::Mat &depth, cv::Mat &depth_cov2);

    /**
     * @brief template matching along epipolar line
     *
     * @param ref                   reference image
     * @param curr                  current image
     * @param T_C_R                 matrix from ref to cur
     * @param pt_ref                point in ref
     * @param depth_mu              mean of depth
     * @param depth_cov             cov of depth
     * @param pt_curr               point in current
     * @param epipolar_direction
     * @return true
     * @return false
     */
    bool epipolarSearch(
        const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R,
        const Eigen::Vector2d &pt_ref, const double &depth_mu, const double &depth_cov,
        Eigen::Vector2d &pt_curr, Eigen::Vector2d &epipolar_direction);

    /**
     * @brief
     *
     * @param pt_ref                point in ref
     * @param pt_curr               point in cur
     * @param T_C_R                 matrix from ref to cur
     * @param epipolar_direction
     * @param depth                 mean of depth
     * @param depth_cov2            cov of depth
     * @return true
     * @return false
     */
    bool updateDepthFilter(
        const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr, const Sophus::SE3d &T_C_R,
        const Eigen::Vector2d &epipolar_direction, cv::Mat &depth, cv::Mat &depth_cov2);

    /**
     * @brief NCC computation(zero mean, normalized)
     *
     * @param ref       reference image
     * @param curr      current image
     * @param pt_ref    referece pixel
     * @param pt_curr   current pixel
     * @return double   NCC score
     */
    double NCC(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr);

    /// Helper functions
    inline double getBilinearInterpolatedValue(const cv::Mat &img, const Eigen::Vector2d &pt)
    {
        uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
        double xx = pt(0, 0) - floor(pt(0, 0));
        double yy = pt(1, 0) - floor(pt(1, 0));
        return ((1 - xx) * (1 - yy) * double(d[0]) +
                xx * (1 - yy) * double(d[1]) +
                (1 - xx) * yy * double(d[img.step]) +
                xx * yy * double(d[img.step + 1])) /
               255.0;
    }

    void plotDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate);

    inline Eigen::Vector3d px2cam(const Eigen::Vector2d px)
    {
        return {
            (px(0, 0) - cx) / fx,
            (px(1, 0) - cy) / fy,
            1};
    }

    inline Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam)
    {
        return {
            p_cam(0, 0) * fx / p_cam(2, 0) + cx,
            p_cam(1, 0) * fy / p_cam(2, 0) + cy};
    };

    inline bool inside(const Eigen::Vector2d &pt)
    {
        return pt(0, 0) >= boarder &&
               pt(1, 0) >= boarder &&
               pt(0, 0) + boarder < width &&
               pt(1, 0) + boarder <= height;
    }

    void showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr,
                           const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_curr);

    void showEpipolarLine(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref,
                          const Eigen::Vector2d &px_min_curr, const Eigen::Vector2d &px_max_curr);

    void evaluateDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate);
};

// Read from dataset
bool readDatasetFiles(
    const std::string &path, int height, int width,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth);

#endif