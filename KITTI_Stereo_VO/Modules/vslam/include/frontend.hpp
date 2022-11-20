#ifndef VSLAM_FRONTEND_HPP
#define VSLAM_FRONTEND_HPP

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace vslam
{
    class Backend;
    class Viewer;

    enum class FrontendStatus
    {
        INITING,
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    /**
     * @brief
     * Estimate the current frame Pose,
     * add keyframes to the map
     * and trigger optimization when the keyframe conditions are met
     */
    class Frontend
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Frontend>;

        Frontend();

        // External interface, add a frame and calculate its positioning result
        bool AddFrame(Frame::Ptr frame);

        // Set function
        inline void SetMap(Map::Ptr map) { map_ = map; }
        inline void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }
        inline void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }
        inline FrontendStatus GetStatus() const { return status_; }
        inline void SetCameras(Camera::Ptr left, Camera::Ptr right)
        {
            camera_left_ = left;
            camera_right_ = right;
        }

    private:
        /**
         * @brief Track in normal mode
         *
         * @return true
         * @return false
         */
        bool Track();

        /**
         * @brief Reset when lost
         *
         * @return true
         * @return false
         */
        bool Reset();

        /**
         * @brief Track with last frame
         *
         * @return num of tracked point
         */
        int TrackLastFrame();

        /**
         * @brief estimate current frame's pose
         *
         * @return num of inliers
         */
        int EstimateCurrentPose();

        /**
         * @brief set current frame as a keyframe and insert it into backend
         *
         * @return true
         * @return false
         */
        bool InsertKeyframe();

        /**
         * @brief Try init the front with stereo images saved in current_frame_
         *
         * @return true
         * @return false
         */
        bool StereoInit();

        /**
         * @brief Detect features in left image in current_frame_
         * keypoints will be saved in current_frame_
         *
         * @return
         */
        int DetectFeatures();

        /**
         * @brief
         * Find the corresponding features in right image of current_frame_
         *
         * @return num of features found
         */
        int FindFeaturesInRight();

        /**
         * @brief Build the initial amp with single image
         *
         * @return true
         * @return false
         */
        bool BuildInitMap();

        /**
         * @brief Triangulate the 2D points in current frame
         *
         * @return num of triangulated points
         */
        int TriangulateNewPoints();

        /**
         * @brief Set the features in keyframe as new observation of the map points
         *
         */
        void SetObservationsForKeyFrame();

        // data
        FrontendStatus status_ = FrontendStatus::INITING;

        Frame::Ptr current_frame_ = nullptr; // current frame
        Frame::Ptr last_frame_ = nullptr;    // Last frame
        Camera::Ptr camera_left_ = nullptr;  // Left camera
        Camera::Ptr camera_right_ = nullptr; // right camera

        Map::Ptr map_ = nullptr;
        std::shared_ptr<Backend> backend_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;

        // The relative motion between current frame and previous frame is used to estimate the initial value of the current frame pose
        SE3d relative_motion_;

        // inliers, used for testing new keyframes
        int tracking_inliers_ = 0;

        // params
        int num_features_ = 200;
        int num_features_init_ = 100;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;

        // utilities
        cv::Ptr<cv::GFTTDetector> gftt_; // feature detector in opencv
    };
}

#endif
