get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
add_library(mxfp8_pack STATIC pack.cpp)
target_compile_options(mxfp8_pack PRIVATE -O2)
target_compile_features(mxfp8_pack PRIVATE cxx_std_20)
target_include_directories(mxfp8_pack PRIVATE
  "${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")

add_library(mxfp8_pack_ant STATIC EXCLUDE_FROM_ALL pack.cpp)
target_link_libraries(mxfp8_pack_ant PRIVATE runtime_ant_tasks)
target_compile_options(mxfp8_pack_ant PRIVATE -O2)
target_compile_features(mxfp8_pack_ant PRIVATE cxx_std_20)
target_include_directories(mxfp8_pack_ant PRIVATE
  "${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")
