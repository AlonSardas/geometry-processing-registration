#ifndef TEST_MESHES_H
#define TEST_MESHES_H

#include <Eigen/Core>

inline void generate_circle_mesh(
    Eigen::MatrixXd &V, Eigen::MatrixXi &F,
    const Eigen::RowVector3d &circle_center = Eigen::RowVector3d::Zero(),
    double circle_radius = 1.0, int num_segments = 100) {
  V.resize(num_segments + 1, 3);
  F.resize(num_segments, 3);

  V.row(0) = circle_center;
  for (int i = 0; i < num_segments; ++i) {
    double angle = 2.0 * M_PI * i / num_segments;
    V.row(i + 1) =
        circle_center +
        circle_radius * Eigen::RowVector3d(cos(angle), sin(angle), 0.0);
  }

  for (int i = 0; i < num_segments - 1; ++i) {
    F.row(i) = Eigen::RowVector3i(0, i + 1, i + 2);
  }
  F.row(num_segments - 1) = Eigen::RowVector3i(0, num_segments, 1);
}

inline void generate_cube_mesh(
    Eigen::MatrixXd &V, Eigen::MatrixXi &F,
    const Eigen::RowVector3d &center = Eigen::RowVector3d::Zero(),
    double size = 1.0) {
  double h = size / 2.0;
  V.resize(8, 3);
  F.resize(12, 3);

  // 8 vertices
  V << center(0) - h, center(1) - h, center(2) - h, // 0: left  bottom back
      center(0) + h, center(1) - h, center(2) - h,  // 1: right bottom back
      center(0) + h, center(1) + h, center(2) - h,  // 2: right top    back
      center(0) - h, center(1) + h, center(2) - h,  // 3: left  top    back
      center(0) - h, center(1) - h, center(2) + h,  // 4: left  bottom front
      center(0) + h, center(1) - h, center(2) + h,  // 5: right bottom front
      center(0) + h, center(1) + h, center(2) + h,  // 6: right top    front
      center(0) - h, center(1) + h, center(2) + h;  // 7: left  top    front

  // 12 triangles (2 per face), consistent winding order (outward normals)
  F << 0, 2, 1,         // back
      0, 3, 2, 4, 5, 6, // front
      4, 6, 7, 0, 1, 5, // bottom
      0, 5, 4, 3, 6, 2, // top
      3, 7, 6, 0, 4, 7, // left
      0, 7, 3, 1, 2, 6, // right
      1, 6, 5;
}

#endif