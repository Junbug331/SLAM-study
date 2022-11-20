#ifndef VSLAM_ALGORITHM_HPP
#define VSLAM_ALGORITHM_HPP

#include "common_include.hpp"

namespace vslam
{
    /**
     * @brief Linear triangulation with SVD
     *
     * @param poses         poses
     * @param points        points in normalized plane
     * @param pt_world      triangulated point in the world
     * @return true
     * @return false
     *
     */
    inline bool triangulation(const std::vector<SE3d> &poses,
                              const std::vector<Vec3> points,
                              Vec3 &pt_world)
    {
        MatXX A(2 * poses.size(), 4);
        VecX b(2 * poses.size());
        b.setZero();

        for (size_t i = 0; i < poses.size(); ++i)
        {
            Mat34 m = poses[i].matrix3x4();
            A.block<1, 4>(i * 2, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(i * 2 + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2)
        {
            // The solution quality is not good, give up
            return true;
        }
        return false;
    }

    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return {p.x, p.y}; }
}

#endif