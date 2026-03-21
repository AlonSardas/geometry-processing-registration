#include "random_points_on_mesh.h"
#include "test_meshes.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

TEST(MeshRandomPointsTest, TestSingleTriangle) {
  int n = 100;
  Eigen::MatrixXd V(3, 3);
  V << 0, 0, 0, //
      1, 0, 0,  //
      0, 2, 0;
  Eigen::MatrixXi F(1, 3);
  F << 0, 1, 2;
  Eigen::MatrixXd X;

  random_points_on_mesh(n, V, F, X);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(X(i, 2), 0.0, 1e-6);
    EXPECT_GE(X(i, 0), 0.0);
    EXPECT_GE(X(i, 1), 0.0);
    EXPECT_LE(X(i, 1), 2 - 2 * X(i, 0));
  }
}

TEST(MeshRandomPointsTest, TestPlanarCircleMesh) {
  Eigen::RowVector3d circle_center(10, 12, 15);
  double circle_radius = 2.2;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  generate_circle_mesh(V, F, circle_center, circle_radius, 100);

  int n = 300;
  Eigen::MatrixXd X;

  random_points_on_mesh(n, V, F, X);

  for (int i = 0; i < n; ++i) {
    EXPECT_LE((X.row(i) - circle_center).norm(), circle_radius);
  }
}

TEST(MeshRandomPointsTest, TestTwoTriangles) {
  int n = 500;
  Eigen::MatrixXd V(4, 3);
  V << -0.5, 0, 0, //
      0, 0.5, 0,   //
      0, 2, 0,     //
      0, -1, 0;
  Eigen::MatrixXi F(2, 3);
  F << 0, 1, 2, //
      1, 3, 0;
  Eigen::MatrixXd X;

  random_points_on_mesh(n, V, F, X);

  int triangle1_count = 0, triangle2_count = 0;
  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(X(i, 2), 0, 1e-6) << "z-component must be zero";
    if (X(i, 1) > 0) {
      triangle1_count += 1;
    } else {
      triangle2_count += 1;
    }
  }

  EXPECT_NEAR(static_cast<double>(triangle1_count) / n, 2.0 / 3, 0.1);
  EXPECT_NEAR(static_cast<double>(triangle2_count) / n, 1.0 / 3, 0.1);
}
