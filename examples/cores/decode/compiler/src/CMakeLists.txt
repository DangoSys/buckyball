if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

message(STATUS "Building Decode compiler Core package")
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

set(DECODE_BALL_COMPILER_DIALECT_DIRS
  ${BB_REPO_ROOT}/examples/balls/toint8/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/int2fp/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/vector/compiler/src/Dialect/Buckyball
)

set(BB_BALL_BACKEND_INTRINSICS_DIRS)
foreach(DECODE_BALL_COMPILER_DIALECT_DIR ${DECODE_BALL_COMPILER_DIALECT_DIRS})
  if(NOT EXISTS ${DECODE_BALL_COMPILER_DIALECT_DIR})
    message(FATAL_ERROR
      "Missing decode ball compiler dialect dir: ${DECODE_BALL_COMPILER_DIALECT_DIR}")
  endif()
  include_directories(${DECODE_BALL_COMPILER_DIALECT_DIR})
  get_filename_component(DECODE_BALL_COMPILER_DIR
    ${DECODE_BALL_COMPILER_DIALECT_DIR}/../..
    ABSOLUTE)
  list(APPEND BB_BALL_BACKEND_INTRINSICS_DIRS
    ${DECODE_BALL_COMPILER_DIR}/Backend/include)
endforeach()

set(BB_BALL_BACKEND_INTRINSICS_DIRS ${BB_BALL_BACKEND_INTRINSICS_DIRS}
  CACHE INTERNAL "Buckyball ball backend intrinsic directories")

add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)

add_custom_target(buckyball-compiler-${BB_CORE_NAME}
  DEPENDS
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyball
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyballTransforms
  ${BB_CORE_TARGET_PREFIX}BuddyBuckyballToLLVMIRTranslation
  ${BB_CORE_TARGET_PREFIX}LowerBuckyballPass
  ${BB_CORE_TARGET_PREFIX}LowerTileToBuckyballPass
)
