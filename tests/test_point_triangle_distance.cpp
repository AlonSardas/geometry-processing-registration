#include "point_triangle_distance.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

TEST(PointTriangleDistanceTest, TestInsideTriangle) {
  Eigen::RowVector3d x(0.1, 0.3, 0);
  Eigen::RowVector3d a(1, -1, 0);
  Eigen::RowVector3d b(0, 1, 0);
  Eigen::RowVector3d c(-1, -1, 0);
  double d;
  Eigen::RowVector3d p;

  point_triangle_distance(x, a, b, c, d, p);

  ASSERT_NEAR(d, 0, 1e-6) << "Distance of point inside triangle should be 0";
  EXPECT_NEAR(p(0), x(0), 1e-6);
  EXPECT_NEAR(p(1), x(1), 1e-6);
  EXPECT_NEAR(p(2), 0.0, 1e-6)
      << "Projected point is not in the triangle plane";
}

TEST(PointTriangleDistanceTest, TestAboveTriangle) {
  Eigen::RowVector3d x(0.1, 0.3, 4.6);
  Eigen::RowVector3d a(1, -1, 0);
  Eigen::RowVector3d b(0, 1, 0);
  Eigen::RowVector3d c(-1, -1, 0);
  double d;
  Eigen::RowVector3d p;

  point_triangle_distance(x, a, b, c, d, p);

  ASSERT_NEAR(d, x(2), 1e-6)
      << "Distance of point above triangle should be the z-component";
  EXPECT_NEAR(p(0), x(0), 1e-6);
  EXPECT_NEAR(p(1), x(1), 1e-6);
  EXPECT_NEAR(p(2), 0.0, 1e-6)
      << "Projected point is not in the triangle plane";
}

TEST(PointTriangleDistanceTest, TestClosestIsVertex) {
  Eigen::RowVector3d x(0.1, 6.0, 0.2);
  Eigen::RowVector3d a(1, -1, 0);
  Eigen::RowVector3d b(0, 1, 0);
  Eigen::RowVector3d c(-1, -1, 0);
  double d;
  Eigen::RowVector3d p;

  point_triangle_distance(x, a, b, c, d, p);

  ASSERT_NEAR(d, (x - b).norm(), 1e-6)
      << "Point distance should be the distance from the triangle vertex";
  EXPECT_NEAR(p(0), b(0), 1e-6);
  EXPECT_NEAR(p(1), b(1), 1e-6);
  EXPECT_NEAR(p(2), b(2), 1e-6);
}

TEST(PointTriangleDistanceTest, TestClosestIsEdge) {
  Eigen::RowVector3d x(0.3, -6.0, 0.5);
  Eigen::RowVector3d a(1, -1, 0);
  Eigen::RowVector3d b(0, 1, 0);
  Eigen::RowVector3d c(-1, -1, 0);
  double d;
  Eigen::RowVector3d p;

  point_triangle_distance(x, a, b, c, d, p);

  ASSERT_NEAR(d, std::sqrt(std::pow(5.0, 2) + std::pow(0.5, 2)), 1e-6)
      << "Distance should be the distance to the triangle edge";
  EXPECT_NEAR(p(0), 0.3, 1e-6);
  EXPECT_NEAR(p(1), -1.0, 1e-6);
  EXPECT_NEAR(p(2), 0.0, 1e-6);
}
