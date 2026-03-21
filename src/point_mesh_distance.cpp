#include "point_mesh_distance.h"
#include "point_triangle_distance.h"
#include <igl/per_face_normals.h>

void point_mesh_distance(const Eigen::MatrixXd &X, const Eigen::MatrixXd &VY,
                         const Eigen::MatrixXi &FY, Eigen::VectorXd &D,
                         Eigen::MatrixXd &P, Eigen::MatrixXd &N) {
  assert(X.cols() == 3 && "X must have 3 columns");
  assert(VY.cols() == 3 && "VY must have 3 columns");
  assert(FY.cols() == 3 && "FY must have 3 columns");

  P.resizeLike(X);
  N.resizeLike(X);
  D.resize(X.rows());

  // The most naive implementation of just iterating all the triangles for each
  // query point in X
  // Consider using AABB instead
  Eigen::MatrixXd triangles_normals;
  igl::per_face_normals(VY, FY, triangles_normals);
  double distance;
  Eigen::RowVector3d p;
  for (int i = 0; i < X.rows(); ++i) {
    int best_triangle = 0;
    D(i) = std::numeric_limits<double>::infinity();

    for (int triangle_index = 0; triangle_index < FY.rows(); ++triangle_index) {
      point_triangle_distance(X.row(i), VY.row(FY(triangle_index, 0)),
                              VY.row(FY(triangle_index, 1)),
                              VY.row(FY(triangle_index, 2)), distance, p);
      if (distance < D(i)) {
        D(i) = distance;
        P.row(i) = p;
        best_triangle = triangle_index;
      }
    }
    N.row(i) = triangles_normals.row(best_triangle);
  }
}
