cmake_minimum_required(VERSION 3.10)
project(bebop)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")
link_directories("$ENV{RISCV}/lib")
set(CMAKE_INSTALL_RPATH $ENV{RISCV}/lib)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

add_library(bebop SHARED
  src/bebop.cc
  src/socket/socket.cc
  src/socket/socket_cmd.cc
  src/socket/socket_dma.cc
)


target_include_directories(bebop PRIVATE
  $ENV{RISCV}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(bebop PROPERTIES OUTPUT_NAME "bebop")

install(TARGETS bebop
  LIBRARY DESTINATION ${CMAKE_INSTALL_RPATH}
)
