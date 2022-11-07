#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

namespace fs = std::experimental::filesystem;
using namespace Eigen;
using TrajectoryType = std::vector<Sophus::SE3d>;
using std::cout;
using std::endl;

void ReadTrajectory(TrajectoryType &trajs, const std::string &path);
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

int main() {
  fs::path root_dir(ROOT_DIR);
  fs::path gt_path = root_dir / "groundtruth.txt";
  fs::path esti_path = root_dir / "estimated.txt";
  TrajectoryType gt, esti;
  ReadTrajectory(gt, gt_path.string());
  ReadTrajectory(esti, esti_path.string());
  assert(!gt.empty() && !esti.empty());
  assert(gt.size() == esti.size());

  // compute rmse
  double rmse = 0;
  for (std::size_t i = 0; i < esti.size(); ++i) {
    Sophus::SE3d p1 = esti[i], p2 = gt[i];
    double error = (p2.inverse() * p1).log().norm();
    rmse += error * error;
  }
  rmse /= static_cast<double>(esti.size());
  rmse = sqrt(rmse);
  cout << "RMSE = " << rmse << endl;

  DrawTrajectory(gt, esti);

  return 0;
}

void ReadTrajectory(TrajectoryType &trajs, const std::string &path) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    std::cerr << "file can't open\n";
    exit(1);
  }

  trajs.clear();
  trajs.reserve(1000);

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    trajs.emplace_back(
        Sophus::SE3d(Quaterniond(qx, qy, qz, qw), Vector3d(tx, ty, tz)));
  }

}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
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

    glLineWidth(2) ;
    for (std::size_t i=0; i<gt.size()-1; ++i)
    {
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINES);
        auto p1 = gt[i].translation(), p2 = gt[i+1].translation();
        glVertex3d(p1[0], p1[1], p1[2]);
        glVertex3d(p2[0], p2[1], p2[2]);
        glEnd();
    }

    for (std::size_t i=0; i<esti.size()-1; ++i)
    {
        glColor3f(1.0f, 0.0f, 0.0f);
        glBegin(GL_LINES);
        auto p1 = esti[i].translation(), p2 = esti[i+1].translation();
        glVertex3d(p1[0], p1[1], p1[2]);
        glVertex3d(p2[0], p2[1], p2[2]);
        glEnd();
    }

    pangolin::FinishFrame();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}