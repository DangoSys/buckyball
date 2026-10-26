# Host compiler plugin for this design's core.
set(ENGINE "${DESIGN_REPO}/stack/engine")
set(TARGET ${DESIGN_CORE_TARGET_${CORE}})

add_subdirectory("${ENGINE}/compiler" engine-compiler)
add_library(engine-compiler-plugin MODULE $<TARGET_OBJECTS:engine-compiler>)
set_target_properties(engine-compiler-plugin PROPERTIES OUTPUT_NAME engine-compiler
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
