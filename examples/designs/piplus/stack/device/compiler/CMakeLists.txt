# ===----------------------------------------------------------------------===
# The compiler of the piplus device: the framework midend (the dialects and
# lower-device, which lowers the launches and transfers to the job runtime and
# the piplus device runtime) and the piplus pass that places a program on the
# cores, as the buddy-opt plugin libdevice-compiler.so, which compiles the
# device program. The graphs are compiled by the compiler of the cores,
# core/compiler.
#
# Built by the stack top of the device simulation for the host, configured
# with DESIGN_CMAKE. Passes (src/):
#   convert-linalg-to-device  launches on the cores of their class, transfers
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-device-compiler LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(BUCKYBALL_BALLS)
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" midend)
add_library(device-compiler MODULE)
target_link_libraries(device-compiler PRIVATE buckyball-compiler)
set_target_properties(device-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
