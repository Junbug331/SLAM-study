#ifndef VSLAM_CAMERA_HPP
#define VSLAM_CAMERA_HPP

#include "common_include.hpp"

namespace vslam
{
    // pinhole stereo camera model
    class Camera
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        using Ptr = std::shared_ptr<Camera>;

        // Camera intrinsics
        double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
        double baseline_ = 0.0;
        SE3d pose_;     // extrinsic, from steteo camera to single camera
        SE3d pose_inv_; // inverse of extrinsics

        Camera() = default;
        Camera(double fx, double fy, double cx, double cy, double baseline, const SE3d &pose)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose)
        {
            pose_inv_ = pose_.inverse();
        }

        inline SE3d pose() const { return pose_; }

        // return intrinsic matrix
        inline Mat33 K() const
        {
            Mat33 k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0.0, 0.0, 1.0;
            return k;
        }

        // coordinate transform: world, camera, pixel
        inline Vec3 world2camera(const Vec3 &p_w, const SE3d &T_c_w)
        {
            return pose_ * T_c_w * p_w;
        }

        inline Vec3 camera2world(const Vec3 &p_c, const SE3d &T_c_w)
        {
            return T_c_w.inverse() * pose_inv_ * p_c;
        }

        inline Vec2 camera2pixel(const Vec3 &p_c)
        {
            return Vec2(
                fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
                fy_ * p_c(1, 0) / p_c(2, 0) + cy_);
        }

        inline Vec3 pixel2camera(const Vec2 &p_p, double depth = 1)
        {
            return Vec3(
                (p_p(0, 0) - cx_) * depth / fx_,
                (p_p(1, 0) - cy_) * depth / fy_,
                depth);
        }

        inline Vec3 pixel2world(const Vec2 &p_p, const SE3d &T_c_w, double depth = 1)
        {
            return camera2world(pixel2camera(p_p, depth), T_c_w);
        }

        inline Vec2 world2pixel(const Vec3 &p_w, const SE3d &T_c_w)
        {
            return camera2pixel(world2camera(p_w, T_c_w));
        }
    };

}

#endif