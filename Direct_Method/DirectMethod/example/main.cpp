#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cassert>

#include "direct_method.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

int main()
{
    fs::path res_dir(RES_DIR);
    fs::path left_file = res_dir / "left.png";
    fs::path disparity_file = res_dir / "disparity.png";

    Mat left_img = imread(left_file, 0);
    Mat disparity_img = imread(disparity_file, 0);
    assert(!left_img.empty() || !disparity_img.empty());

    int n_points = 2000;
    int boarder = 20;
    // Camera intrinsics
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // baseline
    double baseline = 0.573;

    // generate pixels in ref and load depth data
    DirectMethod dm(left_img, disparity_img, n_points, boarder, fx, fy, cx, cy, baseline);

    // estimates 01~05.png's pose using thie information
    Sophus::SE3d T_cur_ref;

    int pyramids = 4;
    double pyramid_scale = 0.5;
    for (int i = 1; i <= 5; ++i)
    {
        fs::path img_path = res_dir / string_format("%06d.png", i);
        Mat img = imread(img_path, 0);
        std::cout << img_path << std::endl;
        // dm.DirectPoseEstimationSingleLayer(img, T_cur_ref, true);
        dm.DirectPoseEstimationMultiLayer(img, T_cur_ref, pyramids, pyramid_scale, true);
    }

    return 0;
}
