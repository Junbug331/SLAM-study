#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

/*
Rotation Matrix(3 x 3): Eigen::Matrix3d

Rotation Vector(3 x 1): Eigen::AngleAxisd

Euler Angle(3 x 1): Eigen::Vector3d

Quaternion(4 x 1): Eigen::Quaterniond

Euclidean tranformation matrix(4 x 4): Eigen::Isometry3d
Affine transformation(4 x 4): Eigen::Affine3d
Perspective transformation(4 x 4): Eigen::Projective3d
*/

int main(int argc, char* argv[])
{
    // 3D rotation matrix(SO3)
    Matrix3d rotation_matrix = Matrix3d::Identity(); 

    // Angle Axis(so3)
    AngleAxisd rotation_vector(M_PI/4, Vector3d(0, 0, 1)); // Rotate 45 deg, along z-axis

    std::cout.precision(3);
    std::cout << "rotation matrix = \n" << rotation_vector.matrix() << std::endl; // convert to matrix
    std::cout << std::endl;

    // assign to rotation matrix directly
    rotation_matrix = rotation_vector.toRotationMatrix();

    // coordinate transformation with AngleAxis
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    std::cout << "(1, 0, 0) after roation (by angle axis) = " << v_rotated.transpose() << std::endl;

    // Or use a rotation matrix
    v_rotated = rotation_matrix * v;
    std::cout << "(1, 0, 0) after roation (by Rotation Matrix) = " << v_rotated.transpose() << std::endl;
    std::cout << std::endl;

    // Euler angle
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX order, roll->pitch->yaw
    std::cout << "yaw pitch roll = " << euler_angles.transpose() << std::endl << std::endl;

    // Euclidean transformation matrix using Eigen::Isometry
    Isometry3d T = Isometry3d::Identity(); // essentially 4x4 matrix
    T.rotate(rotation_vector); // Rotate according to rotation_vector
    T.pretranslate(Vector3d(1, 3, 4)); // Set the translation vector to (1, 3, 4)
    std::cout << "Transform matrix = \n" << T.matrix() << std::endl << std::endl;

    // Use Transformation matrix for coordinate transformation
    Vector3d v_transformed = T * v; // R*v + t
    std::cout << "v transformed = " << v_transformed.transpose() << std::endl << std::endl;

    // For affine and projective transformations, use Eigen::Affine3d and Eigen::Projective3d
    // Quaternion
    Quaterniond q = Quaterniond(rotation_vector);
    std::cout << "quaternion from rotation vector = " << q.coeffs().transpose() << std::endl << std::endl;

    // Rotate a vector with a quaternion and use overloaded multiplication
    v_rotated = q * v; // q(theta/2) * v * q^{-1}(theta/2)
    std::cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << std::endl << std::endl;
    std::cout << " equal to q * v * q^-1 = " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << std::endl << std::endl;


    return 0;
}