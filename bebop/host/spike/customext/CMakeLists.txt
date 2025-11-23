cmake_minimum_required(VERSION 3.10)
project(bebop)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")
set(SPIKE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../riscv-isa-sim")
set(SPIKE_PREFIX "${SPIKE_ROOT}/install")

set(SPIKE_INCLUDE "${SPIKE_PREFIX}/include")
set(SPIKE_LIB_DIR "${SPIKE_PREFIX}/lib")

link_directories("${SPIKE_LIB_DIR}")
set(CMAKE_INSTALL_RPATH "${SPIKE_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

set(BEBOP_INSTALL_LIB_DIR "${SPIKE_LIB_DIR}")

add_library(bebop SHARED
  src/bebop.cc
)

if(NOT TARGET bebop_ipc)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ipc
                   ${CMAKE_CURRENT_BINARY_DIR}/ipc)
endif()


target_include_directories(bebop PRIVATE
  ${SPIKE_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bebop PRIVATE bebop_ipc)

set_target_properties(bebop PROPERTIES OUTPUT_NAME "bebop")

install(TARGETS bebop
  LIBRARY DESTINATION ${BEBOP_INSTALL_LIB_DIR}
)
