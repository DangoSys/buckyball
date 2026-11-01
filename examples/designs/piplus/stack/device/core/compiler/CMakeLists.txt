# ===----------------------------------------------------------------------===
# The compiler of the piplus cores: the framework midend (the dialects,
# lower-core, which lowers the launches to batches of asynchronous engine
# jobs, and lower-frontend-fake-op, which computes the remaining frontend
# operations of the graphs in software on the core CPU) and the piplus pass
# that places the parts of a graph its core's hardware thread can compute on
# that thread, as the buddy-opt plugin libcore-compiler.so.
#
# Built by the stack top of the device simulation for the host, configured
# with DESIGN_CMAKE. Passes (src/):
#   convert-linalg-to-core  MXINT8 quantization and matrix products on the
#                           thread, as subgraphs launched as early as their
#                           inputs allow; transfers to and from it
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-core-compiler LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(BUCKYBALL_BALLS)
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" midend)
add_library(core-compiler MODULE)
target_link_libraries(core-compiler PRIVATE buckyball-compiler)
set_target_properties(core-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
