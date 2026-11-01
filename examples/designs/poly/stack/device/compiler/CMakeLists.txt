# ===----------------------------------------------------------------------===
# The compiler of the poly device: the framework midend (the dialects,
# lower-device, which lowers the launches and transfers to the job runtime and
# the poly device runtime, and lower-frontend-fake-op) and the poly pass that
# places a program on the cores, as the buddy-opt plugin
# libdevice-compiler.so. It compiles the device program and the graphs of the
# reduce core, which has no hardware thread; the graphs of the poly cores go
# through the core compiler (core/compiler).
#
# Built by the stack top of the device simulation for the host, configured
# with DESIGN_CMAKE. Passes (src/):
#   convert-linalg-to-device  launches on the cores of their class, transfers
#                             over the core links
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device-compiler LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(BUCKYBALL_BALLS)
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" midend)
add_library(device-compiler MODULE)
target_link_libraries(device-compiler PRIVATE buckyball-compiler)
set_target_properties(device-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
