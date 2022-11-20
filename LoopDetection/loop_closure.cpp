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
    fs::path vocab_path = res_dir / "vocabulary.yml.gz";
    // read the iamges and database
    spdlog::info("reading database...");
    DBoW3::Vocabulary vocab(vocab_path);

    if (vocab.empty())
    {
        cerr << "Vocabulary does not exit\n";
        return 1;
    }

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

    // We can compare the images directly or we can compare one image to a database
    spdlog::info("comparing images with images");
    for (int i = 0; i < images.size(); ++i)
    {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = 0; j < images.size(); ++j)
        {
            if (i == j)
                continue;
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);
            spdlog::info("image{} vs image{} : {}", i, j, score);
        }
        cout << endl;
    }

    // or with database
    spdlog::info("comparing images with database");
    DBoW3::Database db(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); ++i)
        db.add(descriptors[i]);
    cout << "database info: " << db << endl;
    for (int i = 0; i < descriptors.size(); ++i)
    {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4); // max result = 4
        cout << "searching for image " << i << " returns " << ret << endl
             << endl;
    }
    spdlog::info("done");

    return 0;
}