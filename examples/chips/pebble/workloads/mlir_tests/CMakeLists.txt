add_custom_target(pebble-mlir-tests-build)

add_subdirectory(bank)
add_subdirectory(tile)
add_subdirectory(linalg)

add_dependencies(chip-mlirtests-build pebble-mlir-tests-build)
