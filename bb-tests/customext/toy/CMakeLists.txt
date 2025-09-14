project(toy)

# Check if RISCV environment variable is set
# if(NOT DEFINED ENV{RISCV})
#   message(FATAL_ERROR "RISCV is unset")
# else()
#   message(STATUS "Running with RISCV=$ENV{RISCV}")
# endif()

# # Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

# Add library directories and rpath
link_directories(${BSPIKE_DIR}/lib)
set(CMAKE_INSTALL_RPATH ${BSPIKE_DIR}/lib)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Create shared library
add_library(toy SHARED
  toy.cc
  diff/ref.cc
  inst/24_mvin.cc
  inst/25_mvout.cc
  inst/26_bbfp_mul.cc
  inst/32_mul_warp16.cc
  inst/33_im2col.cc
  inst/34_transpose.cc
)

# Add include directories (modern CMake approach)
target_include_directories(toy PRIVATE
  ${BSPIKE_DIR}/include
  ${TOY_EXT_DIR}
  ${TOY_EXT_DIR}/include
  ${TOY_EXT_DIR}/diff
  ${WORKLOADS_DIR}/lib
)

# Link against required libraries
target_link_libraries(toy PRIVATE
  riscv
  disasm
  softfloat
  dl
)

# 添加依赖，确保bbhw-x86在toy之前构建，并链接生成的库文件
add_dependencies(toy bbhw-x86)
target_link_libraries(toy PRIVATE ${CMAKE_BINARY_DIR}/workloads/lib/bbhw/libbbhw-x86.a)

# Set output name to match Makefile
set_target_properties(toy PROPERTIES OUTPUT_NAME "toy")

# Install target
install(TARGETS toy
  LIBRARY DESTINATION CMAKE_INSTALL_RPATH
)
