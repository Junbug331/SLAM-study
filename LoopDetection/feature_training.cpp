#include <iostream>
#include <vector>
#include <string>
#include <experimental/filesystem>

#include <DBoW3/DBoW3.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <spdlog/spdlog.h>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

int main()
{
    fs::path res_dir(RES_DIR);

    // read image
    spdlog::info("reading images...");
    vector<Mat> images;
    for (int i = 0; i < 10; ++i)
    {
        fs::path img_path = res_dir / (to_string(i + 1) + ".png");
        images.emplace_back(imread(img_path));
    }

    // detect ORB features
    spdlog::info("detecting ORB features...");
    Ptr<ORB> detector = ORB::create();
    vector<Mat> descriptors(images.size());
    for (int i = 0; i < images.size(); ++i)
    {
        vector<KeyPoint> kpts;
        detector->detectAndCompute(images[i], noArray(), kpts, descriptors[i]);
    }

    // create vocabulary
    spdlog::info("creating vocabulary...");
    // Default constructor (k = 10, L(depth) = 5)
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save(res_dir / "vocabulary.yml.gz");
    spdlog::info("done");

    return 0;
}