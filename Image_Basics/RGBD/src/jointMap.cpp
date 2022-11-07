#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

namespace fs = std::experimental::filesystem;
using TrajectoryType = std::vector<Sophus::SE3d>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

template <typename... Args>
std::string string_format(const std::string &format, Args... args) {
  size_t size =
      snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

void showPointCloud(const std::vector<Vector6d> &point_cloud);

void createPointCloud_naive(const std::vector<cv::Mat> &color_imgs,
                            const std::vector<cv::Mat> &depth_imgs,
                            const TrajectoryType &poses, float fx, float fy,
                            float cx, float cy, float depth_scale,
                            std::vector<Vector6d> &point_cloud);

void createPointCloud_opecvParallel(const std::vector<cv::Mat> &color_imgs,
                                    const std::vector<cv::Mat> &depth_imgs,
                                    const TrajectoryType &poses, float fx,
                                    float fy, float cx, float cy,
                                    float depth_scale,
                                    std::vector<Vector6d> &point_cloud);

void createPointCloud_Threads(const std::vector<cv::Mat> &color_imgs,
                              const std::vector<cv::Mat> &depth_imgs,
                              const TrajectoryType &poses, float fx, float fy,
                              float cx, float cy, float depth_scale,
                              std::vector<Vector6d> &point_cloud);

void createPointCloud_async(const std::vector<cv::Mat> &color_imgs,
                            const std::vector<cv::Mat> &depth_imgs,
                            const TrajectoryType &poses, float fx, float fy,
                            float cx, float cy, float depth_scale,
                            std::vector<Vector6d> &point_cloud);

int main() {
  fs::path pose_path = fs::path(RES_DIR) / "pose.txt";
  fs::path color_dir = fs::path(RES_DIR) / "color";
  fs::path depth_dir = fs::path(RES_DIR) / "depth";

  std::vector<cv::Mat> color_imgs, depth_imgs;
  TrajectoryType poses;

  std::ifstream fin(pose_path);
  if (!fin.is_open()) {
    std::cerr << "can't open pose.txt\n";
    return EXIT_FAILURE;
  }

  for (int i = 1; i <= 5; ++i) {
    std::string color_img_path = color_dir / (std::to_string(i) + ".png");
    std::string depth_img_path = depth_dir / (std::to_string(i) + ".pgm");
    color_imgs.emplace_back(cv::imread(color_img_path));
    depth_imgs.emplace_back(
        cv::imread(depth_img_path, -1)); // use -1 flag to load the depth image

    double data[7] = {0};
    for (double &d : data)
      fin >> d;
    poses.emplace_back(
        Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                     Eigen::Vector3d(data[0], data[1], data[2])));
  }

  // Compute the point cloud using camera intrinsics
  double cx = 325.5;
  double cy = 253.5;
  double fx = 518.0;
  double fy = 519.0;
  double depth_scale = 1000.0;
  std::vector<Vector6d> point_cloud;
  point_cloud.reserve(1000000);

  /// Naive
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  createPointCloud_naive(color_imgs, depth_imgs, poses, fx, fy, cx, cy,
                         depth_scale, point_cloud);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  double ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "time: " << ts / 1000.0 << std::endl;
  point_cloud.clear();

  // Only Opencv-Parallel
  begin = std::chrono::steady_clock::now();
  createPointCloud_opecvParallel(color_imgs, depth_imgs, poses, fx, fy, cx, cy,
                                 depth_scale, point_cloud);
  end = std::chrono::steady_clock::now();
  ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "(opencv-parallel) time: " << ts / 1000.0 << std::endl;
  point_cloud.clear();

  /// Multi-thread
  begin = std::chrono::steady_clock::now();
  createPointCloud_Threads(color_imgs, depth_imgs, poses, fx, fy, cx, cy,
                           depth_scale, point_cloud);
  end = std::chrono::steady_clock::now();
  ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "(multi-thread) time: " << ts / 1000.0 << std::endl;

  point_cloud.clear();

  /// Future
  begin = std::chrono::steady_clock::now();
  createPointCloud_async(color_imgs, depth_imgs, poses, fx, fy, cx, cy,
                         depth_scale, point_cloud);
  end = std::chrono::steady_clock::now();
  ts =
      (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();
  std::cout << "(future) time: " << ts / 1000.0 << std::endl;

   showPointCloud(point_cloud);

  return 0;
}

void showPointCloud(const std::vector<Vector6d> &pointcloud) {

  if (pointcloud.empty()) {
    std::cerr << "Point cloud is empty!" << std::endl;
    return;
  }

  pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175),
                                         1.0, -1024.0f / 768.0f)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto &p : pointcloud) {
      glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
      glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();
    pangolin::FinishFrame();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return;
}

void createPointCloud_naive(const std::vector<cv::Mat> &color_imgs,
                            const std::vector<cv::Mat> &depth_imgs,
                            const TrajectoryType &poses, float fx, float fy,
                            float cx, float cy, float depth_scale,
                            std::vector<Vector6d> &point_cloud) {
  for (int i = 0; i < 5; i++) {
    cv::Mat color = color_imgs[i];
    cv::Mat depth = depth_imgs[i];
    Sophus::SE3d T = poses[i];
    for (int v = 0; v < color.rows; v++)
      for (int u = 0; u < color.cols; u++) {
        unsigned int d = depth.ptr<unsigned short>(v)[u];
        if (d == 0)
          continue;
        Eigen::Vector3d point;
        point[2] = double(d) / depth_scale;
        point[0] = (u - cx) * point[2] / fx;
        point[1] = (v - cy) * point[2] / fy;
        Eigen::Vector3d pointWorld = T * point;

        Vector6d p;
        p.head<3>() = pointWorld;
        p[5] = color.data[v * color.step + u * color.channels()];     // blue
        p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
        p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
        point_cloud.push_back(p);
      }
  }
}

void createPointCloud_opecvParallel(const std::vector<cv::Mat> &color_imgs,
                                    const std::vector<cv::Mat> &depth_imgs,
                                    const TrajectoryType &poses, float fx,
                                    float fy, float cx, float cy,
                                    float depth_scale,
                                    std::vector<Vector6d> &point_cloud) {

  for (int i = 0; i < 5; ++i) {
    cv::Mat color = color_imgs[i];
    cv::Mat depth = depth_imgs[i];
    Sophus::SE3d T = poses[i];
    cv::parallel_for_(
        cv::Range(0, color.rows * color.cols), [&](const cv::Range &range) {
          for (int r = range.start; r < range.end; ++r) {
            int v = r / color.cols;
            int u = r - v * color.cols;
            unsigned int d =
                depth.ptr<unsigned short>(v)[u]; // depth value is 16-bit
            if (d == 0)
              continue;
            Eigen::Vector3d point;
            point[2] = double(d) / depth_scale;
            point[0] = (u - cx) * point[2] / fx;
            point[1] = (v - cy) * point[2] / fy;
            Eigen::Vector3d point_world = T * point;

            Vector6d p;
            p.head<3>() = point_world;
            p[5] = color.data[v * color.step + u * color.channels()]; // blue
            p[4] =
                color.data[v * color.step + u * color.channels() + 1]; // green
            p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
            point_cloud.push_back(p);
          }
        });
  }
}

void createPointCloud_Threads(const std::vector<cv::Mat> &color_imgs,
                              const std::vector<cv::Mat> &depth_imgs,
                              const TrajectoryType &poses, float fx, float fy,
                              float cx, float cy, float depth_scale,
                              std::vector<Vector6d> &point_cloud) {
  std::vector<std::thread> threads;
  threads.reserve(5);
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back(std::thread(
        [&](int idx) {
          cv::Mat color = color_imgs[idx];
          cv::Mat depth = depth_imgs[idx];
          Sophus::SE3d T = poses[idx];
          cv::parallel_for_(
              cv::Range(0, color.rows * color.cols),
              [&](const cv::Range &range) {
                for (int r = range.start; r < range.end; ++r) {
                  int v = r / color.cols;
                  int u = r - v * color.cols;
                  unsigned int d =
                      depth.ptr<unsigned short>(v)[u]; // depth value is 16-bit
                  if (d == 0)
                    continue;
                  Eigen::Vector3d point;
                  point[2] = double(d) / depth_scale;
                  point[0] = (u - cx) * point[2] / fx;
                  point[1] = (v - cy) * point[2] / fy;
                  Eigen::Vector3d point_world = T * point;

                  Vector6d p;
                  p.head<3>() = point_world;
                  p[5] =
                      color.data[v * color.step + u * color.channels()]; // blue
                  p[4] = color.data[v * color.step + u * color.channels() +
                                    1]; // green
                  p[3] = color.data[v * color.step + u * color.channels() +
                                    2]; // red
                  point_cloud.push_back(p);
                }
              });
        },
        i));
  }

  for (int i = 0; i < threads.size(); ++i)
    threads[i].join();
}

void createPointCloud_async(const std::vector<cv::Mat> &color_imgs,
                            const std::vector<cv::Mat> &depth_imgs,
                            const TrajectoryType &poses, float fx, float fy,
                            float cx, float cy, float depth_scale,
                            std::vector<Vector6d> &point_cloud) {

  std::vector<std::future<void>> futures;
  futures.reserve(5);
  for (int i = 0; i < 5; ++i) {
    futures.emplace_back(std::async(
        [&](int idx) {
          cv::Mat color = color_imgs[idx];
          cv::Mat depth = depth_imgs[idx];
          Sophus::SE3d T = poses[idx];
          cv::parallel_for_(
              cv::Range(0, color.rows * color.cols),
              [&](const cv::Range &range) {
                for (int r = range.start; r < range.end; ++r) {
                  int v = r / color.cols;
                  int u = r - v * color.cols;
                  unsigned int d =
                      depth.ptr<unsigned short>(v)[u]; // depth value is 16-bit
                  if (d == 0)
                    continue;
                  Eigen::Vector3d point;
                  point[2] = double(d) / depth_scale;
                  point[0] = (u - cx) * point[2] / fx;
                  point[1] = (v - cy) * point[2] / fy;
                  Eigen::Vector3d point_world = T * point;

                  Vector6d p;
                  p.head<3>() = point_world;
                  p[5] =
                      color.data[v * color.step + u * color.channels()]; // blue
                  p[4] = color.data[v * color.step + u * color.channels() +
                                    1]; // green
                  p[3] = color.data[v * color.step + u * color.channels() +
                                    2]; // red
                  point_cloud.push_back(p);
                }
              });
        },
        i));
  }

  for (int i = 0; i < futures.size(); ++i)
    futures[i].get();
}
