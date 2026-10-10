cmake_minimum_required(VERSION 3.12)
project(BuckyballWorkloads C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(common/CMakeLists.txt)

add_subdirectory(${WORKLOAD_LIB_DIR} ${CMAKE_CURRENT_BINARY_DIR}/lib)
add_subdirectory(src)

add_custom_target(sync-bin DEPENDS sync-native-bin sync-mlirtest-bin sync-soctest-bin)
add_custom_target(build-all ALL DEPENDS sync-bin)
