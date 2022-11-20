#pragma once
#ifndef MAP_H
#define MAP_H

#include "common_include.hpp"
#include "frame.hpp"
#include "mappoint.hpp"

namespace vslam
{
    /**
     * @brief Interaction with the map:
     *        the front end calls InsertKeyframe and InsertMapPoint to insert new frames and map points,
     *        the back end maintains the structure of the map, determines outlier/elimination, etc.
     */
    class Map
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Map>;
        using LandmarksType = std::unordered_map<unsigned long, MapPoint::Ptr>;
        using KeyFramesType = std::unordered_map<unsigned long, Frame::Ptr>;

        Map() = default;

        // Add a keyframe
        void InsertKeyFrame(Frame::Ptr frame);

        // Add a map vertex
        void InsertMapPoint(MapPoint::Ptr map_point);

        // Get all map points
        inline LandmarksType GetAllMapPoints()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return landmarks_;
        }

        // Get all keyframes
        inline KeyFramesType GetAllKeyFrames()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;
        }

        // Get the active map point
        inline LandmarksType GetActiveMapPoints()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_landmarks_;
        }

        // Get the active keyframes
        inline KeyFramesType GetActiveKeyFrames()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_keyframes_;
        }

        // Clear the points in the map where the number of observations is zero
        void CleanMap();

    private:
        void RemoveOldKeyframe();

    private:
        std::mutex data_mutex_;
        LandmarksType landmarks_;        // all landmarks
        LandmarksType active_landmarks_; // active landmarks
        KeyFramesType keyframes_;        // all keyframes
        KeyFramesType active_keyframes_; // active keyframes

        Frame::Ptr current_frame_ = nullptr;

        // settings
        int num_active_keyframes_ = 7; // Number of active keyframes
    };
}

#endif