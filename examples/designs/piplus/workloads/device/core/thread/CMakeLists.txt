# Every core with a hardware thread runs thread_sum on its hart: the sum
# program is built once per compiler target and bound to the thread's signature.
set(ENGINE_DIR ${BUCKYBALL_REPO_ROOT}/stack/engine)
foreach(HART IN LISTS DESIGN_DEVICE_CORES)
    set(THREADS ${DESIGN_DEVICE_CORES_${HART}_THREAD_INDICES})
    list(LENGTH THREADS THREAD_COUNT)
    if(THREAD_COUNT EQUAL 0)
        continue()
    endif()
    if(NOT THREAD_COUNT EQUAL 1)
        message(FATAL_ERROR "thread_sum runs on a core with one thread; hart ${HART} has ${THREAD_COUNT}")
    endif()
    set(PROFILE ${DESIGN_THREADS_${THREADS}_PROFILE})
    set(PROGRAM_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROFILE})
    if(NOT TARGET ${PROFILE}-sum-program)
        add_custom_command(
            OUTPUT ${PROGRAM_DIR}/sum.elf ${PROGRAM_DIR}/sum.bin
            COMMAND ${CMAKE_COMMAND} -E make_directory ${PROGRAM_DIR}
            COMMAND
                ${ELF_CC} -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
                -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
                -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
                -nostdlib -nostartfiles -Wl,--no-relax
                -Wl,-T,${ENGINE_DIR}/program/code.ld
                ${ENGINE_DIR}/program/start.S ${CMAKE_CURRENT_SOURCE_DIR}/sum.c
                -o ${PROGRAM_DIR}/sum.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${PROGRAM_DIR}/sum.elf ${PROGRAM_DIR}/sum.bin
            DEPENDS
                sum.c sum.h
                ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
            VERBATIM
        )
        add_custom_target(${PROFILE}-sum-program DEPENDS ${PROGRAM_DIR}/sum.bin)
    endif()
    set(SUM "${DESIGN_NAME}-core${HART}-ctest-${PROFILE}_thread_sum")
    set(SUM_ELF ${CMAKE_CURRENT_BINARY_DIR}/${SUM}-baremetal)
    add_custom_command(
        OUTPUT ${SUM_ELF}
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREADS}_SIGNATURE}ULL -I${ENGINE_DIR}/include
            --embed-dir=${PROGRAM_DIR}
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/thread_sum.c
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
            -lgcc -o ${SUM_ELF}
        DEPENDS
            thread_sum.c sum_binding.c sum.h ${PROGRAM_DIR}/sum.bin
            runtime-baremetal engine
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${SUM}-baremetal-build DEPENDS ${SUM_ELF})
    add_dependencies(design-ctests-build ${SUM}-baremetal-build)
endforeach()
