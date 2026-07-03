if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

message(STATUS "Building Pebble compiler chip package")
message(STATUS "  Source: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Binary: ${CMAKE_CURRENT_BINARY_DIR}")

get_filename_component(BB_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../.. ABSOLUTE)
set(BB_COMPILER_COMMON_DIR ${BB_REPO_ROOT}/compiler/src)

if(NOT EXISTS ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/Buckyball.td)
  message(FATAL_ERROR "Missing Buckyball compiler common dir: ${BB_COMPILER_COMMON_DIR}")
endif()

include_directories(${BB_COMPILER_COMMON_DIR})
include_directories(${BB_COMPILER_COMMON_DIR}/Dialect)
include_directories(${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Dialect)
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/Dialect/Buckyball)

set(PEBBLE_BALL_COMPILER_DIALECT_DIRS
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/systolicarray/compiler/src/Dialect/Buckyball
)

foreach(PEBBLE_BALL_COMPILER_DIALECT_DIR ${PEBBLE_BALL_COMPILER_DIALECT_DIRS})
  if(NOT EXISTS ${PEBBLE_BALL_COMPILER_DIALECT_DIR})
    message(FATAL_ERROR
      "Missing pebble ball compiler dialect dir: ${PEBBLE_BALL_COMPILER_DIALECT_DIR}")
  endif()
  include_directories(${PEBBLE_BALL_COMPILER_DIALECT_DIR})
endforeach()

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)

add_custom_target(buckyball-compiler-${BB_CHIP_NAME}
  DEPENDS
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyball
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyballTransforms
  ${BB_CHIP_TARGET_PREFIX}BuddyBuckyballToLLVMIRTranslation
  ${BB_CHIP_TARGET_PREFIX}LowerBuckyballPass
  ${BB_CHIP_TARGET_PREFIX}LowerTileToBuckyballPass
)
