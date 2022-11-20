#include <iostream>
#include <spdlog/spdlog.h>
#include <experimental/filesystem>
#include <glog/logging.h>

#include "visual_odometry.hpp"

using namespace std;
using namespace vslam;
namespace fs = std::experimental::filesystem;

int main()
{
    fs::path config_dir(CONFIG_DIR);
    fs::path config_file_path = config_dir / "default.yaml";
    vslam::VisualOdometry::Ptr vo = make_shared<vslam::VisualOdometry>(config_file_path.string());
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}