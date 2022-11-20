#pragma once
#ifndef VSLAM_FEATURE_HPP
#define VSLAM_FEATURE_HPP

#include <memory>
#include <opencv2/features2d.hpp>
#include "common_include.hpp"

namespace vslam
{
    struct Frame;
    struct MapPoint;

    struct Feature
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Feature>;

        std::weak_ptr<Frame> frame_;        // the frame that takes this feature
        cv::KeyPoint position_;             // 2D pixel position
        std::weak_ptr<MapPoint> map_point_; // assigned map point

        bool is_outlier_ = false;      // is outlier?
        bool is_on_left_image_ = true; // is detected on the left image?

        Feature() = default;
        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame), position_(kp) {}
    };

}

#endif