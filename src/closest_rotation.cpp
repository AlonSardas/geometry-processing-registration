#include "closest_rotation.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

void closest_rotation(const Eigen::Matrix3d &M, Eigen::Matrix3d &R) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::Matrix3d Omega = Eigen::Matrix3d::Identity();
  Omega(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();

  R = svd.matrixU() * Omega * svd.matrixV().transpose();
}
