#include "point_triangle_distance.h"
#include <Eigen/Geometry>

void project_to_plane(const Eigen::RowVector3d &x, const Eigen::RowVector3d &a,
                      const Eigen::RowVector3d &b, const Eigen::RowVector3d &c,
                      double &d, Eigen::RowVector3d &p);

double compute_signed_area(const Eigen::RowVector3d &a,
                           const Eigen::RowVector3d &b,
                           const Eigen::RowVector3d &c,
                           const Eigen::RowVector3d &norm);

/* We compute the distance from the point to the triangle plane and the distance
 * in plane from the projected point in plane to the triangle.
 * For distance in plane, we use barycentric coordinates, see
 * https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Relationship_with_projective_coordinates
 */
void point_triangle_distance(const Eigen::RowVector3d &x,
                             const Eigen::RowVector3d &a,
                             const Eigen::RowVector3d &b,
                             const Eigen::RowVector3d &c, double &d,
                             Eigen::RowVector3d &p) {
  Eigen::RowVector3d plane_normal = (b - a).cross(c - a);
  plane_normal.normalize();
  double distance_to_plane = (x - a).dot(plane_normal);
  Eigen::RowVector3d x_in_plane = (x - distance_to_plane * plane_normal);

  double sarea_ABC = compute_signed_area(a, b, c, plane_normal);
  double sarea_PBC = compute_signed_area(x_in_plane, b, c, plane_normal);
  double sarea_APC = compute_signed_area(a, x_in_plane, c, plane_normal);
  double sarea_ABP = compute_signed_area(a, b, x_in_plane, plane_normal);

  double lambda1 = sarea_PBC / sarea_ABC;
  double lambda2 = sarea_APC / sarea_ABC;
  double lambda3 = sarea_ABP / sarea_ABC;

  double distance_in_plane = std::nan("");
  int positive_count = (lambda1 >= 0) + (lambda2 >= 0) + (lambda3 >= 0);

  if (positive_count == 3) {
    // point is inside triangle
    p = x_in_plane;
    distance_in_plane = 0;
  } else if (positive_count == 2) {
    const Eigen::RowVector3d *vertex1 = nullptr, *vertex2 = nullptr;
    // closest distance is to opposite edge
    if (lambda1 < 0) {
      vertex1 = &b;
      vertex2 = &c;
    } else if (lambda2 < 0) {
      vertex1 = &a;
      vertex2 = &c;
    } else if (lambda3 < 0) {
      vertex1 = &a;
      vertex2 = &b;
    } else {
      assert(false && "Should never reach here: all lambdas are non-negative");
    }

    Eigen::RowVector3d line_dir = *vertex2 - *vertex1;
    double projected_line_length =
        (x_in_plane - *vertex1).dot(line_dir) / line_dir.dot(line_dir);
    // Make sure the projected point is on the line
    projected_line_length = std::clamp(projected_line_length, 0.0, 1.0);
    p = *vertex1 + projected_line_length * line_dir;
    distance_in_plane = (p - x_in_plane).norm();
  } else if (positive_count == 1) {
    const Eigen::RowVector3d *vertex;
    if (lambda1 >= 0) {
      vertex = &a;
    } else if (lambda2 >= 0) {
      vertex = &b;
    } else if (lambda3 >= 0) {
      vertex = &c;
    } else {
      assert(false && "Should never reach here: all lambdas are negative");
    }

    p = *vertex;
    distance_in_plane = (p - x_in_plane).norm();
  } else {
    assert(false && "Should never reach here: all lambdas are negative");
  }

  d = std::sqrt(distance_in_plane * distance_in_plane +
                distance_to_plane * distance_to_plane);
}

void project_to_plane(const Eigen::RowVector3d &x, const Eigen::RowVector3d &a,
                      const Eigen::RowVector3d &b, const Eigen::RowVector3d &c,
                      double &d,

                      Eigen::RowVector3d &p) {
  Eigen::RowVector3d plane_normal = (b - a).cross(c - a);
  plane_normal.normalize();
  d = (x - a).dot(plane_normal);
  p = (x - d * plane_normal);
}

double compute_signed_area(const Eigen::RowVector3d &a,
                           const Eigen::RowVector3d &b,
                           const Eigen::RowVector3d &c,
                           const Eigen::RowVector3d &norm) {
  Eigen::RowVector3d cross = (b - a).cross(c - a);
  double area = cross.norm() / 2;
  int sign = (cross.dot(norm) > 0) ? 1 : -1;
  return sign * area;
}
