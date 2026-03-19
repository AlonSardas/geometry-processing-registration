#include "random_points_on_mesh.h"
#include "random_utils.h"
#include <igl/cumsum.h>
#include <igl/doublearea.h>

void random_points_on_triangle(const Eigen::RowVector3d &v1,
                               const Eigen::RowVector3d &v2,
                               const Eigen::RowVector3d &v3,
                               Eigen::RowVector3d &out);
int get_random_triangle_index(const Eigen::VectorXd &cumsum_vec);

void random_points_on_mesh(const int n, const Eigen::MatrixXd &V,
                           const Eigen::MatrixXi &F, Eigen::MatrixXd &X) {
  assert(X.rows() == n && "X must have n rows");
  assert(X.cols() == 3 && "X must have 3 columns");

  assert(V.cols() == 3 && "V must have 3 columns");
  assert(F.cols() == 3 && "F must have 3 columns");

  int num_of_triangles = F.rows();
  Eigen::VectorXd areas(num_of_triangles);

  igl::doublearea(V, F, areas);
  Eigen::VectorXd cumsum_vec(num_of_triangles);
  igl::cumsum(areas / 2, 1, cumsum_vec);
  cumsum_vec /= cumsum_vec(num_of_triangles - 1);

  Eigen::RowVector3d random_vec;
  for (int i = 0; i < n; ++i) {
    int random_index = get_random_triangle_index(cumsum_vec);
    random_points_on_triangle(V.row(F(random_index, 0)),
                              V.row(F(random_index, 1)),
                              V.row(F(random_index, 2)), random_vec);
    X.row(i) = random_vec;
  }
}

void random_points_on_triangle(const Eigen::RowVector3d &v1,
                               const Eigen::RowVector3d &v2,
                               const Eigen::RowVector3d &v3,
                               Eigen::RowVector3d &out) {
  double a1 = random_double(), a2 = random_double();
  Eigen::RowVector3d v2_v1 = v2 - v1;
  Eigen::RowVector3d v3_v1 = v3 - v1;

  // Ensure the point is inside the triangle
  if (a1 + a2 > 1) {
    a1 = 1 - a1;
    a2 = 1 - a2;
  }

  out = v1 + a1 * v2_v1 + a2 * v3_v1;
}

// This is an alternative implementation to this code
// auto it = std::lower_bound(C.begin(), C.end(), a2);
// int index = std::distance(C.begin(), it);
int get_random_triangle_index(const Eigen::VectorXd &cumsum_vec) {
  double x = random_double();
  int low = 0, high = cumsum_vec.rows() - 1;
  int mid = low + (high - low) / 2;

  while (high - low > 0) {
    if (x > cumsum_vec(mid)) {
      low = mid + 1;
    } else {
      high = mid;
    }
    mid = low + (high - low) / 2;
  }
  return mid;
}
