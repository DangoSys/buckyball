set(STARTUP ${WORKLOAD_LIB_DIR}/bbsw/baremetal/crt0.S)
set(SYSCALLS ${WORKLOAD_LIB_DIR}/bbsw/baremetal/syscalls.c)
set(LINKER_SCRIPT ${WORKLOAD_LIB_DIR}/bbsw/baremetal/bbsim.ld)
add_custom_target(omni-interconnect-build)
set(BUCKYBALL_CTEST_TARGET attention)
set(BUCKYBALL_CTEST_GROUP_TARGET omni-interconnect-build)
add_subdirectory(${BUCKYBALL_REPO_ROOT}/examples/balls/smatmul/workloads/ctests smatmul)
foreach(TEST interconnect reservation backpressure bounds peripherals)
  buckyball_ctest_name(NAME attention chip-${TEST})
  set(BINARY ${NAME}-baremetal)
  add_custom_command(OUTPUT ${BINARY}
    COMMAND ${ELF_CC} -std=gnu11 -O2 -static -march=rv64gc -mcmodel=medany
            -specs=nano.specs -specs=nosys.specs -nostartfiles
            -Wl,-T,${LINKER_SCRIPT}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
            ${STARTUP} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.c ${SYSCALLS} -o ${BINARY}
    DEPENDS ${TEST}.c ${STARTUP} ${SYSCALLS} ${LINKER_SCRIPT}
            ${BUCKYBALL_REPO_ROOT}/stack/runtime/include/interconnect.h
    VERBATIM)
  add_custom_target(${NAME}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BINARY})
  add_dependencies(omni-interconnect-build ${NAME}-build)
endforeach()

buckyball_ctest_name(NAME attention chip-process)
set(BINARY ${NAME}-linux)
add_custom_command(OUTPUT ${BINARY}
  COMMAND ${LINUX_CC} -std=gnu11 -O2 -static -march=rv64gc
          ${CMAKE_CURRENT_SOURCE_DIR}/process.c -o ${BINARY}
  DEPENDS process.c
  VERBATIM)
add_custom_target(${NAME}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BINARY})
add_dependencies(omni-interconnect-build ${NAME}-build)

buckyball_ctest_name(NAME attention chip-host-io)
set(BINARY ${NAME}-linux)
add_custom_command(OUTPUT ${BINARY}
  COMMAND ${LINUX_CC} -std=gnu11 -O2 -static -march=rv64gc
          ${CMAKE_CURRENT_SOURCE_DIR}/host_io.c -o ${BINARY}
  DEPENDS host_io.c VERBATIM)
add_custom_target(${NAME}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BINARY})
add_dependencies(omni-interconnect-build ${NAME}-build)

buckyball_ctest_name(NAME attention chip-mailbox)
set(BINARY ${NAME}-linux)
add_custom_command(OUTPUT ${BINARY}
  COMMAND ${LINUX_CC} -std=gnu11 -O2 -static -march=rv64gc
          -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
          ${CMAKE_CURRENT_SOURCE_DIR}/mailbox.c -o ${BINARY}
  DEPENDS mailbox.c ${BUCKYBALL_REPO_ROOT}/stack/runtime/include/interconnect.h VERBATIM)
add_custom_target(${NAME}-build DEPENDS ${BINARY})
add_dependencies(omni-interconnect-build ${NAME}-build)
