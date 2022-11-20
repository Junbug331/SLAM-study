#pragma once
#ifndef VSLAM_DATASET_HPP
#define VSLAM_DATASET_HPP

#include "common_include.hpp"
#include "camera.hpp"
#include "frame.hpp"

namespace vslam
{
    class Dataset
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Dataset>;
        Dataset(const std::string& dataset_path);

        // initialize the dataset
        bool Init();

        // create and return the next frame containing the stereo images
        Frame::Ptr NextFrame();

        // get camera by id
        inline Camera::Ptr GetCamera(int camera_id) const
        {
            return cameras_.at(camera_id);
        }

    private:
        std::string dataset_path_;
        int current_image_index_ = 0;
        std::vector<Camera::Ptr> cameras_;
    };

}

#endif
