#ifndef BUNDLE_ADJUSTMENT_HPP
#define BUNDLE_ADJUSTMENT_HPP

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "SnavelyReprojection.hpp"
#include "BAL_handler.hpp"

using Vector3d = Eigen::Vector3d;
using Vector2d = Eigen::Vector2d;
using SO3d = Sophus::SO3d;

struct PoseAndIntrinsics
{
    PoseAndIntrinsics() {}

    // set from the given data address
    explicit PoseAndIntrinsics(double *data_addr)
    {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    void set_to(double *data_addr)
    {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i)
            data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i)
            data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    Sophus::SO3d rotation;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0.0;
    double k1 = 0.0, k2 = 0.0;
};

// The vertex of the pose + camera paramterso -> 9 dims
// 0-2: so3, 3-5: translation, 6: focal length, 7-8: radial distortion
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override
    {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double *update) override
    {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    // Project a point based on the estimated value
    Vector2d project(const Vector3d &point)
    {
        // to cam coord
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        // normalized image plane
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return {_estimate.focal * distortion * pc[0],
                _estimate.focal * distortion * pc[1]};
    }

    virtual bool read(std::istream &in) { return true; }
    virtual bool write(std::ostream &out) const { return true; }
};

class VertexPoint : public g2o::BaseVertex<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}
    virtual void setToOriginImpl() override
    {
        _estimate = {0, 0, 0};
    }

    virtual void oplusImpl(const double *update) override
    {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(std::istream &in) { return true; }
    virtual bool write(std::ostream &out) const { return true; }
};

class EdgeProjection : public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override
    {
        auto v0 = (VertexPoseAndIntrinsics *)_vertices[0];
        auto v1 = (VertexPoint *)_vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    /// use numeric derivative
    // Not implementing the Jacobian calculation function.
    // g2o will automatically provide a numerical calculation of Jacobain

    virtual bool read(std::istream &in) { return true; }
    virtual bool write(std::ostream &out) const { return true; }
};

void SolveBA_g2o(BALhandler &bal_problem);

void SolveBA_ceres(BALhandler &bal_problem);

#endif
