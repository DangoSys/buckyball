project(sparse C)

add_subdirectory(ctest)
add_subdirectory(mlir)

add_custom_target(ctest-sparse-build ALL DEPENDS
  sparse-baremetal-build
  VERBATIM)
