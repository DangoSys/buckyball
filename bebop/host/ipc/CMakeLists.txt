set(BEBOP_IPC_SOURCES
  src/socket/socket.cc
  src/socket/socket_cmd.cc
  src/socket/socket_dma.cc
)

add_library(bebop_ipc STATIC ${BEBOP_IPC_SOURCES})

target_include_directories(bebop_ipc
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    $ENV{RISCV}/include
)

set_target_properties(bebop_ipc PROPERTIES
  OUTPUT_NAME "bebop_ipc"
  POSITION_INDEPENDENT_CODE ON
)

set(_bebop_ipc_install_dir ${CMAKE_INSTALL_RPATH})
if(NOT _bebop_ipc_install_dir)
  set(_bebop_ipc_install_dir $ENV{RISCV}/lib)
endif()

install(TARGETS bebop_ipc
  ARCHIVE DESTINATION ${_bebop_ipc_install_dir}
)
