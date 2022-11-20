#include <chrono>
#include "config.hpp"
#include "visual_odometry.hpp"

namespace vslam
{
    VisualOdometry::VisualOdometry(const std::string &config_path)
        : config_file_path_{config_path} {}

    bool VisualOdometry::Init()
    {
        // red from config file
        if (Config::SetParameterFile(config_file_path_) == false)
            return false;

        dataset_ = std::make_shared<Dataset>(Config::Get<std::string>("dataset_dir"));
        CHECK_EQ(dataset_->Init(), true);

        // create components and links
        frontend_ = std::make_shared<Frontend>(); // main thread
        backend_ = std::make_shared<Backend>(); // backend thread constructed
        map_ = std::make_shared<Map>();
        viewer_ = std::make_shared<Viewer>(); // Viewer thread constructed
        // viewer_ = nullptr;

        frontend_->SetBackend(backend_);
        frontend_->SetMap(map_);
        frontend_->SetViewer(viewer_);
        frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

        backend_->SetMap(map_);
        backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
        if (viewer_ != nullptr)
            viewer_->SetMap(map_);

        return true;
    }

    void VisualOdometry::Run()
    {
        while(true)
        {
            LOG(INFO) << "VO is running";
            if (Step() == false)
                break;
        }

        backend_->Stop();
        if (viewer_ != nullptr)
            viewer_->Close();

        LOG(INFO) << "VO exit";
    }

    bool VisualOdometry::Step()
    {
        Frame::Ptr new_frame = dataset_->NextFrame();
        if (new_frame == nullptr) return false;

        auto t1 = std::chrono::steady_clock::now();

        // vo with frame
        bool success = frontend_->AddFrame(new_frame);

        auto t2 = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (t2 - t1).count();
        LOG(INFO) << "VO cost time : " << elapsed << " seconds.";
        return success;
    }
}