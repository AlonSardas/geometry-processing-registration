#include "hausdorff_lower_bound.h"
#include "icp_single_iteration.h"
#include "point_mesh_distance.h"
#include "point_to_plane_rigid_matching.h"
#include "point_to_point_rigid_matching.h"
#include "random_points_on_mesh.h"
#include "test_meshes.h"
#include <Eigen/Geometry>
#include <cmath>
#include <gtest/gtest.h>

// These are very simple tests mainly to make sure the functions don't crush

double compute_distance(const Eigen::MatrixXd &X, const Eigen::MatrixXd &P);
void run_rigid_matching_test(const Eigen::MatrixXd &VX,
                             const Eigen::MatrixXi &FX,
                             const Eigen::MatrixXd &VY,
                             const Eigen::MatrixXi &FY, const int num_samples,
                             const ICPMethod method);

TEST(PointToPointRigidMatching, TestCubeToCircleRigidMatching) {
  Eigen::MatrixXd VX;
  Eigen::MatrixXi FX;
  Eigen::MatrixXd VY;
  Eigen::MatrixXi FY;

  generate_circle_mesh(VX, FX, Eigen::RowVector3d(0, -2, 0), 1.0);
  generate_cube_mesh(VY, FY, Eigen::RowVector3d(6, 0, 0), 4.0);

  int samples = 200;
  run_rigid_matching_test(VX, FX, VY, FY, samples,
                          ICPMethod::ICP_METHOD_POINT_TO_POINT);
}

TEST(PointToPlaneRigidMatching, TestCubeToCircleRigidMatching) {
  Eigen::MatrixXd VX;
  Eigen::MatrixXi FX;
  Eigen::MatrixXd VY;
  Eigen::MatrixXi FY;

  generate_circle_mesh(VX, FX, Eigen::RowVector3d(0, -12, 0), 1.0);
  generate_cube_mesh(VY, FY, Eigen::RowVector3d(-3, 0, 0), 4.0);

  // It is important that the body is not aligned with the axis, because
  // otherwise the matrix A in not invertible.
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 1, 0).normalized());
  VY = (R * VY.transpose()).transpose();

  int samples = 200;
  run_rigid_matching_test(VX, FX, VY, FY, samples,
                          ICPMethod::ICP_METHOD_POINT_TO_PLANE);
}

// This is almost duplicate of icp_single_iteration
void run_rigid_matching_test(const Eigen::MatrixXd &VX,
                             const Eigen::MatrixXi &FX,
                             const Eigen::MatrixXd &VY,
                             const Eigen::MatrixXi &FY, const int num_samples,
                             const ICPMethod method) {
  Eigen::MatrixXd X;
  Eigen::VectorXd D;
  Eigen::MatrixXd P, N;
  random_points_on_mesh(num_samples, VX, FX, X);
  point_mesh_distance(X, VY, FY, D, P, N);

  // averaging the distances using L2 norm
  double initial_distance = D.norm();

  Eigen::Matrix3d R;
  Eigen::RowVector3d t;

  if (method == ICPMethod::ICP_METHOD_POINT_TO_POINT) {
    point_to_point_rigid_matching(X, P, R, t);
  } else if (method == ICPMethod::ICP_METHOD_POINT_TO_PLANE) {
    point_to_plane_rigid_matching(X, P, N, R, t);
  } else {
    assert(false && "Got unknown ICPMethod, shouldn't reach here");
  }

  Eigen::MatrixXd X_transformed = (R * X.transpose()).transpose();
  X_transformed = X_transformed.rowwise() + t;

  double final_distance = compute_distance(X_transformed, P);

  EXPECT_LT(final_distance, initial_distance)
      << "rigid matching did not decrease the distance for method "
      << "point-to-"
      << (method == ICP_METHOD_POINT_TO_PLANE ? "plane" : "point");
}

double compute_distance(const Eigen::MatrixXd &X, const Eigen::MatrixXd &P) {
  return (X - P).norm();
}
