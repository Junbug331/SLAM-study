#pragma once
#ifndef VSLAM_MAPPOINT_HPP
#define VSLAM_MAPPOINT_HPP

#include "common_include.hpp"

namespace vslam
{
    struct Frame;
    struct Feature;

    // landmark
    struct MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<MapPoint>;
        unsigned long id_ = 0; // ID
        bool is_outlier = false;
        Vec3 pos_ = Vec3::Zero(); // Position in world
        std::mutex data_mutex_;
        int observed_times_ = 0; // being observed by feature matching algo
        std::list<std::weak_ptr<Feature>> observations_; // features that observed this map point

        MapPoint() = default;

        MapPoint(long id, Vec3 position);

        inline Vec3 Pos()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        inline void SetPos(const Vec3 &pos)
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        }

        inline void AddObservation(std::shared_ptr<Feature> feature)
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            ++observed_times_;
        }

        void RemoveObservation(std::shared_ptr<Feature> feat);

        std::list<std::weak_ptr<Feature>> GetObs()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };

}

#endif