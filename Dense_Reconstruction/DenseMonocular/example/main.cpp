/*
test_data: http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip
*/

#include "dense_mapping.hpp"
#include <spdlog/spdlog.h>
#include <iostream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace Sophus;
using namespace Eigen;
using namespace cv;

/// parameters
int boarder = 20;        // image borader
int width = 640;         // image width
int height = 480;        // image height
int ncc_window_size = 3; // half window size of NCC
double min_cov = 0.1;    // converge criteria: minimal cov
double max_cov = 10;     // disconverge criteria: maximal cov

double fx = 481.2f; // camera intrinsics
double fy = -480.0f;
double cx = 319.5f;
double cy = 239.5f;

int main()
{
    fs::path root_dir(ROOT_DIR);
    fs::path test_data_dir(RES_DIR);

    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(test_data_dir.string(), height, width,
                                color_image_files, poses_TWC, ref_depth);
    if (!ret)
    {
        cerr << "Reading imag files failed\n";
        return -1;
    }

    cout << "read_total " << color_image_files.size() << " files." << endl;

    Mat ref = imread(color_image_files[0], 0); // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0; // initial value of depth
    double init_cov2 = 3.0;  // inital value of cov

    // rows, cols, type, Scalar
    Mat depth(height, width, CV_64F, init_depth);
    Mat depth_cov2(height, width, CV_64F, init_cov2);

    DenseMono dm(boarder, width, height, ncc_window_size, min_cov, max_cov, fx, fy, cx, cy);

    for (int index = 1; index < color_image_files.size(); ++index)
    {
        spdlog::info("--- loop {} ---", index);
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr)
            continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // T_C_R = T_C_W * T_W_R
        dm.update(ref, curr, pose_T_C_R, depth, depth_cov2);
        dm.evaluateDepth(ref_depth, depth);
        // dm.plotDepth(ref_depth, depth);
        // imshow("image", curr);
        // waitKey(0);
    }

    spdlog::info("estimation returns, saving depth map ...");
    imwrite((root_dir/"depth.png").string(), depth);
    spdlog::info("finished");

    return 0;
}