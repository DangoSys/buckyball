add_custom_target(toy-mlir-tests-build)

add_subdirectory(buckyball)
add_subdirectory(tile)
add_subdirectory(linalg)

add_dependencies(toy-mlir-tests-build
  buckyball-mlir-tests-build
  tile-mlir-tests-build
  linalg-mlir-tests-build)
