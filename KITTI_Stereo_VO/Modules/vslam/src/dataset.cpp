#include <fstream>

#include "dataset.hpp"
#include "frame.hpp"

#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

namespace vslam
{
    Dataset::Dataset(const std::string &dataset_path)
        : dataset_path_(dataset_path) {}

    bool Dataset::Init()
    {
        // read camera intrinsics and extrinsics
        std::ifstream fin(dataset_path_ + "/calib.txt");

        if (!fin)
        {
            std::cerr << "cannot find " << dataset_path_ << "/calib.txt!\n";
            return false;
        }

        // P0: P1: P2: P3:
        for (int i = 0; i < 4; ++i)
        {
            char camera_name[3];
            for (int k = 0; k < 3; ++k)
            {
                // P0: P1: P2: P3:
                fin >> camera_name[k];
            }

            double projection_data[12]; // 3 x 4 rectified projection
            for (int k = 0; k < 12; ++k)
                fin >> projection_data[k];

            Mat33 K;
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];

            Vec3 t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            K = K * 0.5;
            Camera::Ptr new_camare(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2), t.norm(), SE3d(SO3d(), t)));
            cameras_.push_back(new_camare);
        }
        fin.close();
        current_image_index_ = 0;
        return true;
    }

    Frame::Ptr Dataset::NextFrame()
    {
        cv::Mat image_left, image_right;
        // read images
        image_left = cv::imread(string_format("%s/image_%d/%06d.png", dataset_path_.c_str(), 0, current_image_index_), 0);
        image_right = cv::imread(string_format("%s/image_%d/%06d.png", dataset_path_.c_str(), 1, current_image_index_), 0);

        if (image_left.data == nullptr || image_right.data == nullptr)
        {
            LOG(ERROR) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }

        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

        auto new_frame = Frame::CreateFrame();
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
        current_image_index_++;
        return new_frame;
    }
}