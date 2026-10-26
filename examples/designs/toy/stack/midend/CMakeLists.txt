# Host compiler plugin for this design's core.
set(ENGINE "${DESIGN_REPO}/stack/engine")
set(TARGET ${DESIGN_CORE_TARGET_${CORE}})

set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_${TARGET}})
set(BUCKYBALL_TARGET_TABLE "${DESIGN_ISA_DIR}/${TARGET}/target.inc")

add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" compiler)
add_library(${TARGET}-compiler MODULE $<TARGET_OBJECTS:buckyball-compiler>)
set_target_properties(${TARGET}-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
