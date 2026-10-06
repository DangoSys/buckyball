add_custom_target(toy-mlir-tests-build)

add_subdirectory(linalg)

add_dependencies(toy-mlir-tests-build
  linalg-mlir-tests-build)

add_dependencies(chip-mlirtests-build toy-mlir-tests-build)
