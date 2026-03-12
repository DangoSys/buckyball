cmake_minimum_required(VERSION 3.16)
project(pegasus-driver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
add_compile_options(-Wall -Wextra -O2)

# Find libelf (required for ELF parsing)
find_library(LIBELF_LIB elf REQUIRED)
find_path(LIBELF_INCLUDE libelf.h
  PATHS /usr/include /usr/local/include
  REQUIRED)

# ---- pegasus-run executable ----
add_executable(pegasus-run
  src/xdma.cc
  src/scu.cc
  src/uart.cc
  src/elf.cc
  src/main.cc
)

target_include_directories(pegasus-run PRIVATE
  include
  ${LIBELF_INCLUDE}
)

target_link_libraries(pegasus-run PRIVATE
  ${LIBELF_LIB}
)

# Install target (optional)
install(TARGETS pegasus-run DESTINATION bin)
