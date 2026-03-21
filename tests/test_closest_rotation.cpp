#include "closest_rotation.h"
#include <Eigen/Geometry>
#include <gtest/gtest.h>

TEST(ClosestRotationTest, TestValidRotations) {
  srand(42); // fixed seed for reproducibility

  for (int i = 0; i < 50; i++) {
    Eigen::Matrix3d M = Eigen::Matrix3d::Random();
    Eigen::Matrix3d R;
    closest_rotation(M, R);

    // R should be orthogonal: R^T * R = I
    EXPECT_TRUE((R.transpose() * R - Eigen::Matrix3d::Identity()).norm() < 1e-6)
        << "R is not orthogonal for iteration " << i;

    // R should have determinant 1 (not -1, which would be a reflection)
    EXPECT_NEAR(R.determinant(), 1.0, 1e-6)
        << "R does not have determinant 1 for iteration " << i;
  }
}

TEST(ClosestRotationTest, TestRotationRemainsUnchanged) {
  // A valid rotation matrix should remain unchanged
  Eigen::Matrix3d R_input;
  R_input = Eigen::AngleAxisd(
      M_PI / 4, Eigen::Vector3d::UnitZ()); // 45 degree rotation around Z

  Eigen::Matrix3d R_output;
  closest_rotation(R_input, R_output);

  EXPECT_TRUE((R_output - R_input).norm() < 1e-6)
      << "A valid rotation matrix should remain unchanged";
}
