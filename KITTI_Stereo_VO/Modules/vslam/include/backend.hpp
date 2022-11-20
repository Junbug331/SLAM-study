#ifndef VSLAM_BACKEND_HPP
#define VSLAM_BACKEND_HPP

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace vslam
{

    class Backend
    {
    public:
        using Ptr = std::shared_ptr<Backend>;

        // start the backend thread in the constructor
        Backend();

        // Set camera and fetch the params
        inline void SetCameras(Camera::Ptr left, Camera::Ptr right)
        {
            cam_left_ = left;
            cam_right_ = right;
        }

        inline void SetMap(std::shared_ptr<Map> map) { map_ = map; }

        // Optimize and update the map
        void UpdateMap();

        // Stop the backend
        void Stop();

    private:
        // backend thread
        void BackendLoop();

        // optimze the sliding window
        void Optimize(Map::KeyFramesType &keyframes, Map::LandmarksType &landmarks);

    private:
        std::shared_ptr<Map> map_;
        std::thread backend_thread_;
        std::mutex data_mutex_;

        std::condition_variable map_update_;
        std::atomic<bool> backend_running_;

        Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
    };
}

#endif