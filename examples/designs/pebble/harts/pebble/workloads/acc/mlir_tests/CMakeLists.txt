add_custom_target(pebble-mlir-tests-build)

add_subdirectory(bank)
add_subdirectory(hart)
add_subdirectory(linalg)

add_dependencies(design-mlirtests-build pebble-mlir-tests-build)
