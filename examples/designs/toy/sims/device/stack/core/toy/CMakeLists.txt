cmake_minimum_required(VERSION 3.20)
project(toy-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

add_subdirectory("${DESIGN_REPO}/examples/designs/toy/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/examples/designs/toy/stack/midend" midend)
