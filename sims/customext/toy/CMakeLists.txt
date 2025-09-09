project(toy)

# Check if RISCV environment variable is set
if(NOT DEFINED ENV{RISCV})
  message(FATAL_ERROR "RISCV is unset")
else()
  message(STATUS "Running with RISCV=$ENV{RISCV}")
endif()

# # Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

# Add library directories and rpath
link_directories("$ENV{RISCV}/lib")
set(CMAKE_INSTALL_RPATH "$ENV{RISCV}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Create shared library
add_library(toy SHARED
  toy.cc
)

# Add include directories (modern CMake approach)
target_include_directories(toy PRIVATE
  "$ENV{RISCV}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Link against required libraries
target_link_libraries(toy PRIVATE
  riscv
  disasm
  softfloat
)

# Set output name to match Makefile
set_target_properties(toy PROPERTIES OUTPUT_NAME "toy")

# Install target
install(TARGETS toy
  LIBRARY DESTINATION "$ENV{RISCV}/lib"
)