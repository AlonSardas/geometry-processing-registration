#include "point_to_plane_rigid_matching.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

void compute_rotation_matrix_by_axis_angle(const Eigen::Vector3d &a,
                                           Eigen::Matrix3d &R);

void point_to_plane_rigid_matching(const Eigen::MatrixXd &X,
                                   const Eigen::MatrixXd &P,
                                   const Eigen::MatrixXd &N, Eigen::Matrix3d &R,
                                   Eigen::RowVector3d &t) {
  Eigen::MatrixXd X_cross_N_N(X.rows(), 6);
  X_cross_N_N.col(0) =
      X.col(1).cwiseProduct(N.col(2)) - X.col(2).cwiseProduct(N.col(1));
  X_cross_N_N.col(1) =
      X.col(2).cwiseProduct(N.col(0)) - X.col(0).cwiseProduct(N.col(2));
  X_cross_N_N.col(2) =
      X.col(0).cwiseProduct(N.col(1)) - X.col(1).cwiseProduct(N.col(0));
  X_cross_N_N.rightCols(3) = N;

  Eigen::MatrixXd A = X_cross_N_N.transpose() * X_cross_N_N;
  assert(A.rows() == 6 && A.cols() == 6 && "A has wrong size");

  Eigen::VectorXd N_P_minus_X = N.cwiseProduct(P - X).rowwise().sum();
  Eigen::VectorXd b = X_cross_N_N.transpose() * N_P_minus_X;
  assert(b.rows() == 6 && "b has wrong size");

  // A has constant size of 6X6, so this is O(1) in complexity
  if (std::abs(A.determinant()) < 1e-10) {
    std::cout << "A\n" << A << std::endl;
    std::cout << "WARNING: A is not invertible." << std::endl;
  }
  Eigen::VectorXd u = A.inverse() * b;

  Eigen::Vector3d a = u.head(3);

  compute_rotation_matrix_by_axis_angle(a, R);
  t = u.tail(3);
}

void compute_rotation_matrix_by_axis_angle(const Eigen::Vector3d &a,
                                           Eigen::Matrix3d &R) {
  double theta = a.norm();
  if (theta < 1e-10) {
    R = Eigen::Matrix3d::Identity();
    return;
  }
  Eigen::Vector3d w = a / theta;
  Eigen::Matrix3d W;
  W << 0, -w(2), w(1), //
      w(2), 0, -w(0),  //
      -w(1), w(0), 0;
  R = Eigen::Matrix3d::Identity() + std::sin(theta) * W +
      (1 - std::cos(theta)) * W * W;
}
