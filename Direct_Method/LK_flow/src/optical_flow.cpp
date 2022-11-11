#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <experimental/filesystem>
#include <string>
#include <cassert>
#include <vector>
#include <future>
#include <thread>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

namespace fs = std::experimental::filesystem;

// debug
bool multi = false;

// OpticalFlowTracker
class OpticalFlowTracker
{
public:
    OpticalFlowTracker(const cv::Mat &img1_,
                       const cv::Mat &img2_,
                       const std::vector<cv::KeyPoint> &kp1_,
                       std::vector<cv::KeyPoint> &kp2_,
                       std::vector<bool> &success_,
                       bool inverse_ = true, bool has_initial_ = false)
        : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_) {}

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::KeyPoint> &kp1;
    std::vector<cv::KeyPoint> &kp2;
    std::vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/**
 * @brief Single Level optical flow
 *
 * @param img1  first image
 * @param img2  second image
 * @param kp1 keypoints in img1
 * @param kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param success true if a keypoint is tracked successfully
 * @param inverse use inverse formulation
 * @param has_initial
 */
void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse = false,
    bool has_initial = false);

/**
 * @brief multi level optical flow, scale of pyramid is set to 2 by default
 *        the image pyramid will be create inside the function
 *
 * @param img1 the first pyramid
 * @param img2  the second pyramid
 * @param kp1 keypoints in img1
 * @param kp2 keypoints in img2
 * @param success true if a keypoint is tracked successfully
 * @param inverse use inverse formulation
 */
void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse = false);

/**
 * @brief get a gray scale value from reference image (bi-linear interpolation)
 *
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // Boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) +
           xx * (1 - yy) * img.at<uchar>(y, x_a1) +
           (1 - xx) * yy * img.at<uchar>(y_a1, x) +
           xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main()
{
    using namespace cv;
    using namespace std;

    fs::path res_dir(RES_DIR);
    fs::path img1_path = res_dir / "LK1.png";
    fs::path img2_path = res_dir / "LK2.png";

    // images (grayscale)
    Mat img1 = imread(img1_path, 0);
    Mat img2 = imread(img2_path, 0);
    assert(img1.data && img2.data);

    // Key points using GFTT
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // max 500 keypoints
    detector->detect(img1, kp1);

    spdlog::stopwatch sw;

    // track these key points in the second image
    // single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    sw.reset();
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
    spdlog::info("optical flow single level: {}", sw);

    // multilevel LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    sw.reset();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    spdlog::info("optical flow multi level: {}", sw);

    // use OpenCV's flow for validation
    vector<Point2f> pt1, pt2;
    for (const auto &kp : kp1)
        pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    sw.reset();

    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    spdlog::info("optical flow by opencv: {}", sw);

    Mat img2_single;
    cvtColor(img2, img2_single, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); ++i)
    {
        if (success_single[i])
        {
            circle(img2_single, kp2_single[i].pt, 2, Scalar(0, 250, 0), 2);
            line(img2_single, kp1[i].pt, kp2_single[i].pt, Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cvtColor(img2, img2_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); ++i)
    {
        if (success_multi[i])
        {
            circle(img2_multi, kp2_multi[i].pt, 2, Scalar(0, 250, 0), 2);
            line(img2_multi, kp1[i].pt, kp2_multi[i].pt, Scalar(0, 250, 0));
        }
    }

    Mat img2_cv;
    cvtColor(img2, img2_cv, COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); ++i)
    {
        if (status[i])
        {
            circle(img2_cv, pt2[i], 2, Scalar(0, 250, 0), 2);
            line(img2_cv, pt1[i], pt2[i], Scalar(0, 250, 0));
        }
    }

    imshow("tracked single level", img2_single);
    imshow("tracked multi level", img2_multi);
    imshow("tracked opencv", img2_cv);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse,
    bool has_initial)
{
    if (!has_initial)
    {
        kp2.clear();
        kp2.resize(kp1.size());
    }
    success.clear();
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);

    cv::parallel_for_(cv::Range(0, kp1.size()),
                      std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range)
{
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; ++i)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx, dy need to be estimated
        if (has_initial)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0.0, last_cost = 0.0;
        bool succ = true; // indicate if this point succeeded

        // Gaussian-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero(); // Hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero(); // bias
        Eigen::Vector2d J;                           // Jacobian
        for (int iter = 0; iter < iterations; ++iter)
        {
            if (inverse == false)
            {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else
            {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // Compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; ++x)
            {
                for (int y = -half_patch_size; y < half_patch_size; ++y)
                {
                    // I1(x, y) = I2(x + d_x, y + d_y)
                    // e = I1(x, y) - I2(x + d_x, y + d_y)
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    if (inverse == false)
                    {
                        // Jacobian at (x + d_x, y + d_y)
                        // Gradient of Second image at (x + d_y, y + d_y)
                        // This is similar to applying sober filter
                        // [I2_x, I2_y]
                        J = -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                              GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                       0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                              GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    }
                    else if (iter == 0)
                    {
                        // in inverse mode, J keeps same for all iteratons
                        // Note this J does not change when dx, dy is updated
                        J = -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                              GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                       0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                              GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)));
                    }

                    // Compute H, b and set cost
                    b += -error * J; // Gradient Vector of error
                    cost += error * error;
                    if (inverse == false || iter == 0)
                    {
                        // also Update H
                        H += J * J.transpose();
                    }
                }
            }

            // Compute Update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0]))
            {
                // when a block or white patch and H is irreversible
                std::cout << "update is nan\n";
                succ = false;
                break;
            }

            if (iter > 0 && cost > last_cost)
                break;

            // Update dx, dy
            dx += update[0];
            dy += update[1];
            last_cost = cost;
            succ = true;

            if (update.norm() < 1e-2) // converge
                break;
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

void CreateImagePyramidAsync(const cv::Mat &img1,
                             const cv::Mat &img2,
                             int n,
                             double scale,
                             std::vector<cv::Mat> &pyr1, std::vector<cv::Mat> &pyr2)
{
    pyr1.clear();
    pyr2.clear();
    pyr1.resize(n);
    pyr2.resize(n);

    // Copy image to base layer
    cv::copyTo(img1, pyr1[0], cv::noArray());
    cv::copyTo(img2, pyr2[0], cv::noArray());

    // Create scale vector
    std::vector<double> scales(n);
    scales[0] = 1.0;
    for (int i = 1; i < n; ++i)
        scales[i] = scales[i - 1] * scale;

    std::vector<std::future<void>> futures;
    futures.reserve(n);
    for (int i = 1; i < n; ++i)
    {
        futures.emplace_back(std::async(
            std::launch::async, [&](int idx)
            {
            cv::resize(img1, pyr1[idx], cv::Size(img1.cols * scales[idx], img1.rows * scales[idx]));
            cv::resize(img2, pyr2[idx], cv::Size(img2.cols * scales[idx], img2.rows * scales[idx])); },
            i));
    }

    for (int i = 0; i < n - 1; ++i)
        futures[i].get();
}

void CreateImagePyramid(const cv::Mat &img1,
                        const cv::Mat &img2,
                        int pyramids,
                        double pyramid_scale,
                        std::vector<cv::Mat> &pyr1, std::vector<cv::Mat> &pyr2)
{
    pyr1.clear();
    pyr1.reserve(pyramids);
    pyr2.clear();
    pyr2.reserve(pyramids);

    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
}

void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse)
{
    // Parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // Create pyramids
    std::vector<cv::Mat> pyr1, pyr2;
    spdlog::stopwatch sw;
    CreateImagePyramid(img1, img2, pyramids, pyramid_scale, pyr1, pyr2);

    // coarse-to-fine LK tracking in pyramids
    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (const auto &kp : kp1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; --level)
    {
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);

        if (level > 0)
        {
            std::for_each(kp1_pyr.begin(), kp1_pyr.end(), [&](cv::KeyPoint &kp)
                          { kp.pt /= pyramid_scale; });
            std::for_each(kp2_pyr.begin(), kp2_pyr.end(), [&](cv::KeyPoint &kp)
                          { kp.pt /= pyramid_scale; });
        }
    }

    kp2 = std::move(kp2_pyr);
}