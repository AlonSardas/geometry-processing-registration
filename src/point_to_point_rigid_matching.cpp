#include "point_to_point_rigid_matching.h"
#include "closest_rotation.h"
#include <Eigen/Dense>

void point_to_point_rigid_matching(const Eigen::MatrixXd &X,
                                   const Eigen::MatrixXd &P, Eigen::Matrix3d &R,
                                   Eigen::RowVector3d &t) {
  Eigen::RowVector3d p_bar = P.colwise().mean();
  Eigen::RowVector3d x_bar = X.colwise().mean();
  Eigen::MatrixXd X_bar = X.rowwise() - x_bar;
  Eigen::MatrixXd P_bar = P.rowwise() - p_bar;
  assert(X_bar.rows() == X.rows() && X_bar.cols() == 3 &&
         "X_bar has wrong size");
  assert(P_bar.rows() == X.rows() && P_bar.cols() == 3 &&
         "P_bar has wrong size");
  Eigen::Matrix3d M = P_bar.transpose() * X_bar;
  closest_rotation(M, R);
  t = p_bar - x_bar * R.transpose();
}
