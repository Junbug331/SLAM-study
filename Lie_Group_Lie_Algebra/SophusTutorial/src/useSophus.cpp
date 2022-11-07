#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

using std::cout;
using std::endl;
using namespace Eigen;


int main()
{
    // Rotation matrix with 90 deg along Z-axis
    Matrix3d R = AngleAxisd(M_PI/2, Vector3d(0, 0, 1)).toRotationMatrix();
    // or Quaternion
    Quaterniond q(R);

    // Sophus::SO3d can be constructed from rotation matrix
    Sophus::SO3d SO3_R(R);
    // or quaternion
    Sophus::SO3d SO3_q(q);

    // They are equivalent
    cout << "SO(3) from matrix: \n" << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion: \n" << SO3_q.matrix() << endl;
    cout << endl;

    // Use logarithmic map to get the Lie Algebra
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl << endl;

    // hat is from vector to skew-symmetric matrix 
    Matrix3d so3_hat = Sophus::SO3d::hat(so3);
    cout << "so3 hat = \n" << so3_hat << endl << endl;

    // inversely from matrix to vector
    Vector3d so3_vee = Sophus::SO3d::vee(so3_hat);
    cout << "so3 hat vee = " << so3_vee.transpose() << endl << endl;

    // update by perturbation model
    Vector3d update_so3(1e-4, 0, 0); // this is a small update
    // Left perturbation model
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \n" << SO3_updated.matrix() << endl << endl;

    cout << std::string(70, '*') << endl << endl;
    /// Similar for SE(3)
    Vector3d t(1, 0, 0); //translation along X
    Sophus::SE3d SE3_Rt(R, t); // construction SE3 from R, t
    Sophus::SE3d SE3_qt(q, t); // construction SE# from quaternion, t
    cout << "SE3 from R,t = \n" << SE3_Rt.matrix() << endl << endl;
    cout << "SE3 from q,t = \n" << SE3_qt.matrix() << endl << endl;

    // Lie Algebra is 6d vector
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    Vector6d se3 = SE3_qt.log();
    cout << "se3 = " << se3.transpose() << endl << endl;
    // The output shows Sophus puts the translation at first in se(3), then rotation.

    Matrix4d se3_hat = Sophus::SE3d::hat(se3);
    cout << "se3 hat = \n" << se3_hat << endl << endl;
    Vector6d se3_vee = Sophus::SE3d::vee(se3_hat);
    cout << "se3 vee = " << se3_vee.transpose() << endl << endl;

    //Finally the update
    Vector6d update_se3;
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated= = \n" << SE3_updated.matrix() << endl;

    
    return 0;
}