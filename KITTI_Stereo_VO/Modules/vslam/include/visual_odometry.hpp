#pragma once
#ifndef VSLAM_VISUAL_ODOMETRY
#define VSLAM_VISUAL_ODOMETRY

#include "common_include.hpp"
#include "backend.hpp"
#include "frontend.hpp"
#include "dataset.hpp"
#include "viewer.hpp"

namespace vslam
{
    class VisualOdometry
    {
    public:
        using Ptr = std::shared_ptr<VisualOdometry>;

        // Constructor with config file
        VisualOdometry(const std::string &config_path);

        /**
         * @brief Do initialization before run
         *
         * @return true
         * @return false
         */
        bool Init();

        /**
         * @brief start vvo in the dataset
         *
         */
        void Run();

        /**
         * @brief Make a step forward in dataset
         *
         * @return true
         * @return false
         */
        bool Step();

        FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

    private:
        bool inited_ = false;
        std::string config_file_path_;

        Frontend::Ptr frontend_ = nullptr;
        Backend::Ptr backend_ = nullptr;
        Map::Ptr map_ = nullptr;
        Viewer::Ptr viewer_ = nullptr;

        // datatset
        Dataset::Ptr dataset_ = nullptr;
    };

}

#endif