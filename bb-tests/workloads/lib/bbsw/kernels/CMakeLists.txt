get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
add_library(scalar_kernels STATIC mxfp8/pack.cpp mxfp8/quant.cpp)
target_compile_options(scalar_kernels PRIVATE -O2)
target_compile_features(scalar_kernels PRIVATE cxx_std_20)
target_include_directories(scalar_kernels PRIVATE
  "${REPO_ROOT}/stack/compiler/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")
