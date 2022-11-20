#pragma once
#ifndef VSLAM_FRAME_HPP
#define VSLAM_FRAME_HPP

#include "camera.hpp"
#include "common_include.hpp"

namespace vslam
{

    // forward declare
    struct MapPoint;
    struct Feature;

    struct Frame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Frame>;

        unsigned long id_ = 0;          // id of thie frame
        unsigned long keyframe_id_ = 0; // id of key frame
        bool is_keyframe_ = false;      // whether it is a keyframe
        double time_stamp_;             // time stamp
        SE3d pose_;                     // T_cw (world -> cam)
        std::mutex pose_mutex_;         // Pose mutex lock
        Mat left_img_, right_img_;      // Stereo image pair

        // extracted features in left img
        std::vector<std::shared_ptr<Feature>> features_left_;

        // corresponding features in right image, set to nullptr if no corresponding
        std::vector<std::shared_ptr<Feature>> features_right_;

        Frame() = default;
        Frame(long id, double time_stamp, const SE3d &pose, const Mat &left, const Mat &right);

        // set and get pose, thread safe
        inline SE3d Pose()
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        inline void SetPose(const SE3d &pose)
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        // Set the keyframe and assign and keyframe id
        void SetKeyFrame();

        // Factory build mode
        static std::shared_ptr<Frame> CreateFrame();
    };
}

#endif