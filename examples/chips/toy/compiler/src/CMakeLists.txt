if(NOT COMMAND add_mlir_library OR NOT COMMAND add_mlir_dialect)
  message(FATAL_ERROR "MLIR CMake helpers must be loaded. Build as part of buddy-mlir.")
endif()

message(STATUS "Building Toy compiler chip package")
message(STATUS "  Source: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Binary: ${CMAKE_CURRENT_BINARY_DIR}")

get_filename_component(BB_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../.. ABSOLUTE)
set(BB_COMPILER_COMMON_DIR ${BB_REPO_ROOT}/compiler/src)

if(NOT EXISTS ${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball/BuckyballOps.h)
  message(FATAL_ERROR "Missing Buckyball compiler common dir: ${BB_COMPILER_COMMON_DIR}")
endif()

include_directories(${BB_COMPILER_COMMON_DIR})
include_directories(${BB_COMPILER_COMMON_DIR}/Dialect)
include_directories(${BB_COMPILER_COMMON_DIR}/Dialect/Buckyball)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Dialect)
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/Dialect/Buckyball)

set(TOY_BALL_COMPILER_DIALECT_DIRS
  ${BB_REPO_ROOT}/examples/balls/vector/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/relu/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/transpose/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/im2col/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/matrix/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/fp2int/compiler/src/Dialect/Buckyball
  ${BB_REPO_ROOT}/examples/balls/int2fp/compiler/src/Dialect/Buckyball
)

set(BB_CHIP_BACKEND_INTRINSICS_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/Backend/include
  CACHE INTERNAL "Buckyball chip backend intrinsic directory")

set(BB_BALL_BACKEND_INTRINSICS_DIRS)

foreach(TOY_BALL_COMPILER_DIALECT_DIR ${TOY_BALL_COMPILER_DIALECT_DIRS})
  if(NOT EXISTS ${TOY_BALL_COMPILER_DIALECT_DIR})
    message(FATAL_ERROR "Missing toy ball compiler dialect dir: ${TOY_BALL_COMPILER_DIALECT_DIR}")
  endif()
  include_directories(${TOY_BALL_COMPILER_DIALECT_DIR})
  get_filename_component(TOY_BALL_COMPILER_DIR
    ${TOY_BALL_COMPILER_DIALECT_DIR}/../..
    ABSOLUTE)
  list(APPEND BB_BALL_BACKEND_INTRINSICS_DIRS
    ${TOY_BALL_COMPILER_DIR}/Backend/include)
endforeach()

set(BB_BALL_BACKEND_INTRINSICS_DIRS ${BB_BALL_BACKEND_INTRINSICS_DIRS}
  CACHE INTERNAL "Buckyball ball backend intrinsic directories")

# Add subdirectories
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
