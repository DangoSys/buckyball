# Ensure buddy-mlir build dir is available
if(NOT DEFINED ENV{BUDDY_MLIR_BUILD_DIR})
  message(FATAL_ERROR
    "BUDDY_MLIR_BUILD_DIR is not set. Enter `nix develop` shell first.")
endif()

set(OPTEST_BUCKYBALL_DIR ${OPTEST_WORKLOAD_DIR}/buckyball)
set(OPTEST_TILE_DIR ${OPTEST_WORKLOAD_DIR}/tile)
set(OPTEST_LINALG_DIR ${OPTEST_WORKLOAD_DIR}/linalg)

add_subdirectory(buckyball)
add_subdirectory(tile)
add_subdirectory(linalg)

add_custom_target(OpTest-all ALL DEPENDS
    buckyball-OpTest-build
    tile-OpTest-build
    linalg-OpTest-build
)
