# ===----------------------------------------------------------------------===
# The compiler of the poly cores: the framework midend (the dialects,
# lower-core, which lowers the launches to batches of asynchronous engine
# jobs, and lower-frontend-fake-op, which computes the remaining frontend
# operations of the graphs in software on the core CPU) and the poly pass that
# places the parts of a segment graph the hardware threads of a poly core can
# compute on those threads, as the buddy-opt plugin libcore-compiler.so.
#
# Built by the stack top of the device simulation for the host, configured
# with DESIGN_CMAKE. Passes (src/):
#   convert-linalg-to-core  the MXINT8 quantizations and matrix products of a
#                           graph on the attention or ffn threads, as
#                           subgraphs launched as early as their inputs
#                           allow, in turn on the threads of the kind;
#                           transfers to and from them
#   convert-linalg-to-acc   MXINT8 and FP32 Ball mapping for the selected
#                           thread profile, including quantized outputs
# The reduce core has no hardware thread: its graphs have no core level.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-core-compiler LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_attention} ${DESIGN_TARGET_BALL_STACKS_ffn})
list(REMOVE_DUPLICATES BUCKYBALL_BALLS)
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" midend)
add_library(core-compiler MODULE)
target_link_libraries(core-compiler PRIVATE buckyball-compiler)
set_target_properties(core-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
