set(ENGINE_DIR ${BUCKYBALL_REPO_ROOT}/stack/engine)
set(MATMUL_TASK_DIR
    ${BUCKYBALL_REPO_ROOT}/examples/prefab/threads/matmul/stack
)
set(MATMUL_NAME
    "${DESIGN_NAME}-matmul-ctest-design-matmul_threads"
)
set(MATMUL_ELF ${MATMUL_NAME}-baremetal)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MATMUL_ELF}
    COMMAND
        ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${BBSIM_LD}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR}
        -I${ENGINE_DIR}/include -I${MATMUL_TASK_DIR}
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services -I${BAREMETAL_DIR}
        ${BAREMETAL_DIR}/start.S ${CMAKE_CURRENT_SOURCE_DIR}/matmul_threads.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime>
        # How the device's cores run jobs and its hardware threads are loaded.
        $<TARGET_PROPERTY:bindings,INTERFACE_LINK_OPTIONS> $<TARGET_FILE:engine>
        -lgcc -o ${MATMUL_ELF}
    DEPENDS
        matmul_threads.c
        ${MATMUL_TASK_DIR}/matmul.h
        runtime-baremetal
        runtime
        engine
        ${ENGINE_DIR}/include/thread.h
        ${ENGINE_DIR}/include/engine.h
        ${BBSIM_LD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${MATMUL_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MATMUL_ELF}
)
add_dependencies(design-ctests-build ${MATMUL_NAME}-ctest-build)
