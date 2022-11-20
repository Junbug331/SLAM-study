#pragma once
#ifndef BAL_HANDLER_HPP
#define BAL_HANDLER_HPP

#include <string>
#include <memory>

class BALhandler
{
public:
    // load BAL data from text file
    explicit BALhandler(const std::string &filename, bool use_quaternions = false);

    ~BALhandler() = default;

    // save results to text file
    void WriteToFile(const std::string &filename) const;

    // save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    void Normalize();

    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    inline int camera_block_size() const { return use_quaternions_ ? 10 : 9; }
    inline int point_block_size() const { return 3; }

    inline int num_cameras() const { return num_cameras_; }

    inline int num_points() const { return num_points_; }

    inline int num_observations() const { return num_observations_; }

    inline int num_parameters() const { return num_parameters_; }

    inline const int *point_index() const { return point_index_.get(); }
    inline const int *camera_index() const { return camera_index_.get(); }
    inline const double *observations() const { return observations_.get(); }
    inline const double *parameters() const { return parameters_.get(); }
    inline const double *cameras() const { return parameters_.get(); }
    inline const double *points() const { return parameters_.get() + camera_block_size() * num_cameras_; }

    // mutable
    inline double *mutable_cameras() { return parameters_.get(); }
    inline double *mutable_points() { return parameters_.get() + camera_block_size() * num_cameras_; }
    inline double *mutable_camera_for_observation(int i)
    {
        return mutable_cameras() + camera_index_[i] * point_block_size();
    }
    inline double *mutable_point_for_observation(int i)
    {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    inline const double *camera_for_observation(int i) const
    {
        return cameras() + camera_index_[i] * camera_block_size();
    }
    inline const double *point_for_observation(int i) const
    {
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngleAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;
    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

private:
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    std::unique_ptr<int[]> point_index_;
    std::unique_ptr<int[]> camera_index_;
    std::unique_ptr<double[]> observations_;
    std::unique_ptr<double[]> parameters_;
};

#endif