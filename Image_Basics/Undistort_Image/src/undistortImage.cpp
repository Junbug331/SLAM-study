#include <experimental/filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

int main() {
  // Sample Imgae path
  std::string file_name = "distorted.png";
  fs::path img_path = fs::path(RES_DIR) / file_name;

  // rad-tan model parame
  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359,
         p2 = 1.76187114e-05;

  // intrinsics
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  cv::Mat img = cv::imread(img_path, 0); // 0 -> CV_8UC1
  int rows = img.rows, cols = img.cols;
  cv::Mat img_undistort = cv::Mat::zeros({cols, rows}, 0);

  // Compute the pixels in the undistorted one
  cv::parallel_for_(cv::Range(0, rows * cols), [&](const cv::Range &range) {
    for (int idx = range.start; idx < range.end; ++idx) {
      int v = idx / cols;
      int u = idx - v * cols;

      double x = (u - cx) / fx, y = (v - cy) / fy;
      double r = sqrt(x * x + y * y);

      double x_distorted = x * (1 + k1 * pow(r, 2) + k2 * pow(r, 4)) +
                           2 * p1 * x * y + p2 * (pow(r, 2) + 2 * pow(x, 2));
      double y_distorted = y * (1 + k1 * pow(r, 2) + k2 * pow(r, 4)) +
                           p1 * (r * r * 2 * y * y) + 2 * p2 * x * y;

      double u_distorted = fx * x_distorted + cx;
      double v_distorted = fy * y_distorted + cy;

      // Check if the pixel is in the image border
      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols &&
          v_distorted < rows) {
        img_undistort.at<uchar>(v, u) = img.at<uchar>(
            static_cast<int>(v_distorted), static_cast<int>(u_distorted));
      } else {
        img_undistort.at<uchar>(v, u) = 0;
      }
    }
  });

  // show the undistored image
  cv::imshow("distorted", img);
  cv::imshow("undistorted", img_undistort);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}