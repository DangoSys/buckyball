cmake_minimum_required(VERSION 3.20)
project(poly-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

add_subdirectory("${DESIGN_REPO}/examples/designs/poly/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/examples/designs/poly/stack/midend" midend)
