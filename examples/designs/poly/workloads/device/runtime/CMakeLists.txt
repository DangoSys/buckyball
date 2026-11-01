set(DEVICE_TRANSFER_NAME "${DESIGN_NAME}-main-ctest-device_transfer-multicore-baremetal")
set(DEVICE_TRANSFER_DEFINES ${DESIGN_TARGET_DEFS_main})
list(TRANSFORM DEVICE_TRANSFER_DEFINES PREPEND "-D")
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/device_transfer.c")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DEVICE_TRANSFER_NAME}
    COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DEVICE_TRANSFER_DEFINES} ${DESIGN_DEFINES}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-main-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/device_transfer.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc
        -o ${CMAKE_CURRENT_BINARY_DIR}/${DEVICE_TRANSFER_NAME}
    DEPENDS device_transfer.c ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_target(device-transfer-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DEVICE_TRANSFER_NAME})
add_dependencies(poly-ctests-build device-transfer-build)

set(POLY_STREAM_NAME "${DESIGN_NAME}-main-ctest-design-stream")
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/stream_main.c")
add_executable(${POLY_STREAM_NAME}-linux stream_main.c)
# bb_isa.h of the stack comes first: its instructions go into the stream.
target_include_directories(
    ${POLY_STREAM_NAME}-linux
    PRIVATE ${BUCKYBALL_REPO_ROOT}/stack/isa
)
# The Ball instructions of the job go into the attention thread's stream.
target_compile_definitions(
    ${POLY_STREAM_NAME}-linux
    PRIVATE ${DESIGN_DEFS} ${DESIGN_TARGET_DEFS_attention} BUCKYBALL_STREAM
)
target_link_libraries(
    ${POLY_STREAM_NAME}-linux
    PRIVATE bbsw-thread-attention-linux bbsw-device-linux runtime-linux
)
add_dependencies(poly-ctests-build ${POLY_STREAM_NAME}-linux)

buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program.c")
add_executable(poly-main-ctest-program-linux program.c)
target_link_libraries(poly-main-ctest-program-linux PRIVATE runtime-program-linux)
add_dependencies(poly-ctests-build poly-main-ctest-program-linux)

buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/program_transfer.c")
add_executable(poly-main-ctest-program_transfer-linux program_transfer.c)
target_link_libraries(poly-main-ctest-program_transfer-linux PRIVATE poly-runtime-linux)
add_dependencies(poly-ctests-build poly-main-ctest-program_transfer-linux)

buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/heap_tls.c")
set(HEAP_TLS_ELF "${CMAKE_CURRENT_BINARY_DIR}/poly-main-ctest-heap_tls-multicore-baremetal")
set(DEVICE_BOOT "${BUCKYBALL_REPO_ROOT}/stack/runtime/boot")
add_custom_command(
  OUTPUT ${HEAP_TLS_ELF}
  COMMAND ${ELF_CC} -O2 -static -march=rv64gc -mabi=lp64d -mcmodel=medany
          -fno-builtin
          -specs=nano.specs -specs=nosys.specs -nostartfiles
          -Wl,-T,${DEVICE_BOOT}/device.ld ${DESIGN_DEFINES}
          ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/heap_tls.c
          -lgcc -o ${HEAP_TLS_ELF}
  DEPENDS heap_tls.c ${DEVICE_BOOT}/start.S ${DEVICE_BOOT}/syscalls.c ${DEVICE_BOOT}/device.ld
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(heap-tls-build DEPENDS ${HEAP_TLS_ELF})
add_dependencies(poly-ctests-build heap-tls-build)

add_subdirectory(ctests)
