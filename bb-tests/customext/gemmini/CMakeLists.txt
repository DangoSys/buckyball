project(gemmini)

# Check if RISCV environment variable is set
if(NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "RISCV is unset")
else()
  message(STATUS "Running with RISCV=$ENV{RISCV}")
endif()

# # Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

# Add library directories and rpath
link_directories(${BSPIKE_DIR}/lib)
set(CMAKE_INSTALL_RPATH ${BSPIKE_DIR}/lib)
# set(CMAKE_INSTALL_RPATH $ENV{RISCV}/lib)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Create shared library
add_library(gemmini SHARED
  gemmini.cc
)

# Add include directories (modern CMake approach)
target_include_directories(gemmini PRIVATE
  ${BSPIKE_DIR}/include
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Link against required libraries
target_link_libraries(gemmini PRIVATE
  riscv
  disasm
  softfloat
)

# Set output name to match Makefile
set_target_properties(gemmini PROPERTIES OUTPUT_NAME "gemmini")

# Install target
install(TARGETS gemmini
  LIBRARY DESTINATION ${CMAKE_INSTALL_RPATH}
)
