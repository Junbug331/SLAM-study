#ifndef VSLAM_VIEWER_HPP
#define VSLAM_VIEWER_HPP

#include <thread>
#include <pangolin/pangolin.h>

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace vslam
{
    class Viewer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Viewer>;

        Viewer();

        inline void SetMap(Map::Ptr map) { map_ = map; }

        void Close();

        // Add a current frame
        void AddCurrentFrame(Frame::Ptr current_frame);

        // Update the map
        void UpdateMap();

    private:
        void ThreadLoop();

        void DrawFrame(Frame::Ptr frame, const float *color);

        void DrawMapPoints();

        void FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera);

        /// plot the features in current frame into an image
        cv::Mat PlotFrameImage();

    private:
        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::thread viewer_thread_;
        bool viewer_running_ = true;

        std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
        std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
        bool map_updated_ = false;

        std::mutex viewer_data_mutex_;
    };

}

#endif