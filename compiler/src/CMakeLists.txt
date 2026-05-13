# Buckyball External Dialects
# This directory contains Buckyball-specific MLIR dialects maintained
# separately from buddy-mlir and integrated at build time.

if(NOT DEFINED MLIR_MAIN_SRC_DIR)
  message(FATAL_ERROR "MLIR_MAIN_SRC_DIR must be defined. Build as part of buddy-mlir.")
endif()

message(STATUS "Building Buckyball External Dialects")
message(STATUS "  Source: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Binary: ${CMAKE_CURRENT_BINARY_DIR}")

# Include directories for external dialect headers
# This allows #include "Dialect/Buckyball/BuckyballDialect.h" to work
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Also add Dialect subdirectory to include path for #include "Buckyball/BuckyballDialect.h"
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Dialect)

# CRITICAL: Add the generated files directory to include path
# This allows the generated .h.inc files to be found
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/Dialect/Buckyball)

# Add subdirectories
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Target)
