#include "point_mesh_distance.h"
#include "test_meshes.h"
#include <gtest/gtest.h>

TEST(PointMeshDistanceTest, TestInsideCircle) {
  Eigen::RowVector3d circle_center(5, 6, 7);
  double circle_radius = 2.2;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  generate_circle_mesh(V, F, circle_center, circle_radius, 50);

  Eigen::MatrixXd X(4, 3);
  X << 5, 6, 7, //
      5, 8, 7,  //
      4, 6, 7,  //
      4, 5, 7;

  Eigen::VectorXd D;
  Eigen::MatrixXd P;
  Eigen::MatrixXd N;

  point_mesh_distance(X, V, F, D, P, N);

  // All distances should be 0
  for (int i = 0; i < X.rows(); i++) {
    EXPECT_NEAR(D(i), 0.0, 1e-6) << "Distance should be 0 for point " << i;
  }

  // Closest points should equal input points
  for (int i = 0; i < X.rows(); i++) {
    EXPECT_NEAR(P(i, 0), X(i, 0), 1e-6) << "P x mismatch for point " << i;
    EXPECT_NEAR(P(i, 1), X(i, 1), 1e-6) << "P y mismatch for point " << i;
    EXPECT_NEAR(P(i, 2), X(i, 2), 1e-6) << "P z mismatch for point " << i;
  }

  // Normals should all be (0, 0, 1)
  for (int i = 0; i < X.rows(); i++) {
    EXPECT_NEAR(N(i, 0), 0.0, 1e-6) << "N x mismatch for point " << i;
    EXPECT_NEAR(N(i, 1), 0.0, 1e-6) << "N y mismatch for point " << i;
    EXPECT_NEAR(N(i, 2), 1.0, 1e-6) << "N z mismatch for point " << i;
  }
}

TEST(PointMeshDistanceTest, TestAboveCircle) {
  Eigen::RowVector3d circle_center(5, 6, 0);
  double circle_radius = 2.2;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  generate_circle_mesh(V, F, circle_center, circle_radius, 50);

  double z_above = 7;
  Eigen::MatrixXd X(4, 3);
  X << 5, 6, z_above, //
      5, 8, z_above,  //
      4, 6, z_above,  //
      4, 5, z_above;

  Eigen::VectorXd D;
  Eigen::MatrixXd P;
  Eigen::MatrixXd N;

  point_mesh_distance(X, V, F, D, P, N);

  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(D(i), z_above, 1e-6)
        << "Distance should be " << z_above << " for point " << i;
  }

  // Closest points should equal to the projection into the plane
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(P(i, 0), X(i, 0), 1e-6) << "P x mismatch for point " << i;
    EXPECT_NEAR(P(i, 1), X(i, 1), 1e-6) << "P y mismatch for point " << i;
    EXPECT_NEAR(P(i, 2), 0, 1e-6)
        << "P z is not on circle plane for point " << i;
  }

  // Normals should all be (0, 0, 1)
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(N(i, 0), 0.0, 1e-6) << "N x mismatch for point " << i;
    EXPECT_NEAR(N(i, 1), 0.0, 1e-6) << "N y mismatch for point " << i;
    EXPECT_NEAR(N(i, 2), 1.0, 1e-6) << "N z mismatch for point " << i;
  }
}

TEST(PointMeshDistanceTest, TestOutsideCircle) {
  Eigen::RowVector3d circle_center(0, 0, 0);
  double circle_radius = 1.0;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  generate_circle_mesh(V, F, circle_center, circle_radius, 700);

  Eigen::MatrixXd X(3, 3);
  X << 5, 6, 0,  //
      -2, 0, 1,  //
      0, -4, -3; //

  Eigen::VectorXd D;
  Eigen::MatrixXd P;
  Eigen::MatrixXd N;

  point_mesh_distance(X, V, F, D, P, N);

  // The circle is composed from finite edges, so the numerical error is due to
  // this discretization
  double abs_tol = 1e-4;

  for (int i = 0; i < X.rows(); i++) {
    double in_plane_distance = X.row(i).head(2).norm() - circle_radius;
    EXPECT_NEAR(
        D(i),
        std::sqrt(in_plane_distance * in_plane_distance + X(i, 2) * X(i, 2)),
        abs_tol)
        << "Distance from circle is incorrect for point " << i;
  }

  abs_tol = 5e-3;
  // Closest points should equal to the projection onto the circle
  for (int i = 0; i < X.rows(); i++) {
    auto X_row = X.row(i);
    auto projection = X_row.head(2) / X_row.head(2).norm();
    EXPECT_NEAR(P(i, 0), projection(0), abs_tol)
        << "P x mismatch for point " << i;
    EXPECT_NEAR(P(i, 1), projection(1), abs_tol)
        << "P y mismatch for point " << i;
    EXPECT_NEAR(P(i, 2), 0, abs_tol)
        << "P z is not on circle plane for point " << i;
  }

  // Normals should all be (0, 0, 1)
  for (int i = 0; i < X.rows(); i++) {
    EXPECT_NEAR(N(i, 0), 0.0, 1e-6) << "N x mismatch for point " << i;
    EXPECT_NEAR(N(i, 1), 0.0, 1e-6) << "N y mismatch for point " << i;
    EXPECT_NEAR(N(i, 2), 1.0, 1e-6) << "N z mismatch for point " << i;
  }
}

TEST(PointMeshDistanceTest, TestNormalsOnCube) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  auto cube_center = Eigen::RowVector3d::Zero();
  double cube_size = 2;
  generate_cube_mesh(V, F, cube_center, cube_size);

  Eigen::MatrixXd X(4, 3);
  X << 0, 0, 5,      //
      -3, -0.2, 0.8, //
      5, 0.2, -0.7,  //
      0, -2.1, 0.1;  //

  Eigen::VectorXd D;
  Eigen::MatrixXd P;
  Eigen::MatrixXd N;

  point_mesh_distance(X, V, F, D, P, N);

  // Point 0: (0,0,5) -> closest (0,0,1), distance 4, normal (0,0,1)
  EXPECT_NEAR(D(0), 4.0, 1e-6);
  EXPECT_NEAR(P(0, 0), X(0, 0), 1e-6);
  EXPECT_NEAR(P(0, 1), X(0, 1), 1e-6);
  EXPECT_NEAR(P(0, 2), 1.0, 1e-6);
  EXPECT_NEAR(std::abs(N(0, 2)), 1.0, 1e-6);
  EXPECT_NEAR(N(0, 0), 0.0, 1e-6);
  EXPECT_NEAR(N(0, 1), 0.0, 1e-6);

  // Point 1: (-3,-0.2,0.8) -> closest (-1,-0.2,0.8), distance 2, normal
  // (-1,0,0)
  EXPECT_NEAR(D(1), 2.0, 1e-6);
  EXPECT_NEAR(P(1, 0), -1.0, 1e-6);
  EXPECT_NEAR(P(1, 1), X(1, 1), 1e-6);
  EXPECT_NEAR(P(1, 2), X(1, 2), 1e-6);
  EXPECT_NEAR(std::abs(N(1, 0)), 1.0, 1e-6);
  EXPECT_NEAR(N(1, 1), 0.0, 1e-6);
  EXPECT_NEAR(N(1, 2), 0.0, 1e-6);

  // Point 2: (5,0.2,-0.7) -> closest (1,0.2,-0.7), distance 4, normal (1,0,0)
  EXPECT_NEAR(D(2), 4.0, 1e-6);
  EXPECT_NEAR(P(2, 0), 1.0, 1e-6);
  EXPECT_NEAR(P(2, 1), X(2, 1), 1e-6);
  EXPECT_NEAR(P(2, 2), X(2, 2), 1e-6);
  EXPECT_NEAR(std::abs(N(2, 0)), 1.0, 1e-6);
  EXPECT_NEAR(N(2, 1), 0.0, 1e-6);
  EXPECT_NEAR(N(2, 2), 0.0, 1e-6);

  // Point 3: (0,-2.1,0.1) -> closest (0,-1,0.1), distance 1.1, normal (0,-1,0)
  EXPECT_NEAR(D(3), 1.1, 1e-6);
  EXPECT_NEAR(P(3, 0), X(3, 0), 1e-6);
  EXPECT_NEAR(P(3, 1), -1.0, 1e-6);
  EXPECT_NEAR(P(3, 2), X(3, 2), 1e-6);
  EXPECT_NEAR(std::abs(N(3, 1)), 1.0, 1e-6);
  EXPECT_NEAR(N(3, 0), 0.0, 1e-6);
  EXPECT_NEAR(N(3, 2), 0.0, 1e-6);
}

TEST(PointMeshDistanceTest, TestInvalidParams) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  generate_circle_mesh(V, F);

  Eigen::MatrixXd X_bad_cols = Eigen::MatrixXd::Random(4, 5);
  Eigen::VectorXd D;
  Eigen::MatrixXd P;
  Eigen::MatrixXd N;

  ASSERT_DEATH(point_mesh_distance(X_bad_cols, V, F, D, P, N),
               "X must have 3 columns");
}
