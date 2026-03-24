add_executable(run_tests
  test_random_points_on_mesh.cpp
  test_point_triangle_distance.cpp
  test_point_mesh_distance.cpp
  test_hausdorff_lower_bound.cpp
  test_closest_rotation.cpp
  test_rigid_matching.cpp
)

target_link_libraries(run_tests
  core  
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_tests)
