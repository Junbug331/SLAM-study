#pragma once
#ifndef DIRECT_METHOD_HPP
#define DIRECT_METHOD_HPP

#include <vector>
#include <iostream>
#include <algorithm>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix26d = Eigen::Matrix<double, 2, 6>;
using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using VecVector2d = std::vector<Vector2d>;

/// Class for accumulator jacobains in parallel
class JacobianAccumulator
{
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const std::vector<double> &depth_ref_,
        double fx_, double fy_, double cx_, double cy_,
        Sophus::SE3d &T21_);

    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const std::vector<double> &depth_ref_,
        const cv::Mat &K,
        Sophus::SE3d &T21_);

    /* Inline function */
    /// get hessian matrix
    inline Matrix6d hessian() const { return H; }

    /// get bias
    inline Vector6d bias() const { return b; }

    /// get total cost
    inline double cost_func() const { return cost; }

    /// get projected points
    inline VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    inline void reset()
    {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

    /// accumlate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const std::vector<double> &depth_ref;
    double fx, fy, cx, cy;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

class DirectMethod
{
public:
    DirectMethod(const cv::Mat &ref_img_, const cv::Mat &disparity_img_,
                 int n_points_, double boarder_, double fx_, double fy_,
                 double cx_, double cy_, double baseline_);

    void DirectPoseEstimationSingleLayer(
        const cv::Mat &img2, Sophus::SE3d &T_cur_ref, bool vis = false);

    void DirectPoseEstimationMultiLayer(
        const cv::Mat &img2, Sophus::SE3d &T_cur_ref, int n_pyramids, double scale, bool vis = false);

    inline VecVector2d GetPxRef() const { return px_ref; }
    inline std::vector<double> GetDepthRef() const { return depth_ref; }

private:
    static void createImgPyramids(int n_pyramids, double scale, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Mat> &pyr1, std::vector<cv::Mat> &pyr2);
    static void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double> depth_ref,
        double fx, double fy, double cx, double cy,
        int level, /*DEBUG*/
        Sophus::SE3d &T_cur_ref,
        bool vis = false);

private:
    const cv::Mat &ref_img;
    const cv::Mat &disparity_img;
    int n_points = 2000;
    double boarder = 20;
    double fx = 718.856;
    double fy = 718.856;
    double cx = 607.1928;
    double cy = 185.2157;
    double baseline = 0.573;
    VecVector2d px_ref;
    std::vector<double> depth_ref;
};

// Helper Function
inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    // std::clamp(x, 0.f, float(img.cols - 1));
    // std::clamp(y, 0.f, float(img.rows - 1));
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols)
        x = img.cols - 1;
    if (y >= img.rows)
        y = img.rows - 1;

    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}

#endif