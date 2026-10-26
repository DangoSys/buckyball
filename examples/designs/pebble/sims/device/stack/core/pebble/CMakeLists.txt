cmake_minimum_required(VERSION 3.20)
project(pebble-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

add_subdirectory("${DESIGN_REPO}/examples/designs/pebble/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/examples/designs/pebble/stack/midend" midend)
