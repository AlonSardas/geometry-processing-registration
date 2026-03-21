#include "hausdorff_lower_bound.h"
#include "test_meshes.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(HausdorffLowerBoundTest, TestCircleCubeDistance) {
  Eigen::MatrixXd VX;
  Eigen::MatrixXi FX;
  Eigen::MatrixXd VY;
  Eigen::MatrixXi FY;

  generate_circle_mesh(VX, FX, Eigen::RowVector3d(0, 0, 0), 1.0);
  generate_cube_mesh(VY, FY, Eigen::RowVector3d(6, 0, 0), 4.0);

  int samples = 300;
  double circle_cube_distance = hausdorff_lower_bound(VX, FX, VY, FY, samples);
  double cube_circle_distance = hausdorff_lower_bound(VY, FY, VX, FX, samples);
  EXPECT_NEAR(circle_cube_distance, 5.0, 1e-1);
  // Farthest point is on the corner, which is quite hard to sample randomly, so
  // we just make sure it's within reasonable bounds
  double real_cube_circle_distance = std::sqrt(7 * 7 + 2 * 2 + 2 * 2);
  EXPECT_LE(cube_circle_distance, real_cube_circle_distance);
  EXPECT_GE(cube_circle_distance, real_cube_circle_distance - 0.5);
}

TEST(HausdorffLowerBoundTest, TestZeroDistanceOfSubCircle) {
  Eigen::MatrixXd VX;
  Eigen::MatrixXi FX;
  Eigen::MatrixXd VY;
  Eigen::MatrixXi FY;

  generate_circle_mesh(VX, FX, Eigen::RowVector3d(0, 0, 0), 5.0);
  generate_circle_mesh(VY, FY, Eigen::RowVector3d(1, 0, 0), 1.0);

  int samples = 400;
  double outer_inner_distance = hausdorff_lower_bound(VX, FX, VY, FY, samples);
  double inner_outer_distance = hausdorff_lower_bound(VY, FY, VX, FX, samples);
  EXPECT_NEAR(outer_inner_distance, 5.0, 2e-1);
  EXPECT_NEAR(inner_outer_distance, 0, 1e-6);
}
