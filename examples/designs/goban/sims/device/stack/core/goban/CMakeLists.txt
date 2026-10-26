cmake_minimum_required(VERSION 3.20)
project(goban-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

add_subdirectory("${DESIGN_REPO}/examples/designs/goban/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/examples/designs/goban/stack/midend" midend)
