# Every action hart runs thread_sum on its hart: the sum program is bound to
# the signature of its action threads (all of one kind).
set(KERNEL_DIR ${BUCKYBALL_REPO_ROOT}/stack/kernel)
foreach(HART IN LISTS DESIGN_DEVICE_HARTS)
    set(THREADS ${DESIGN_DEVICE_HARTS_${HART}_THREAD_INDICES})
    list(LENGTH THREADS THREAD_COUNT)
    if(THREAD_COUNT EQUAL 0)
        continue()
    endif()
    list(GET THREADS 0 FIRST)
    set(PROFILE ${DESIGN_THREADS_${FIRST}_PROFILE})
    if(NOT PROFILE STREQUAL "action")
        continue()
    endif()
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
                -Wl,-T,${KERNEL_DIR}/program/code.ld
                ${KERNEL_DIR}/program/start.S ${CMAKE_CURRENT_SOURCE_DIR}/sum.c
                -o ${PROGRAM_DIR}/sum.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${PROGRAM_DIR}/sum.elf ${PROGRAM_DIR}/sum.bin
            DEPENDS
                sum.c sum.h
                ${KERNEL_DIR}/program/start.S ${KERNEL_DIR}/program/code.ld
            VERBATIM
        )
        add_custom_target(${PROFILE}-sum-program DEPENDS ${PROGRAM_DIR}/sum.bin)
    endif()
    set(SUM "${DESIGN_NAME}-hart${HART}-ctest-${PROFILE}_thread_sum")
    set(SUM_ELF ${CMAKE_CURRENT_BINARY_DIR}/${SUM}-baremetal)
    add_custom_command(
        OUTPUT ${SUM_ELF}
        COMMAND
            ${ELF_CC} ${CTEST_C_FLAGS}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${FIRST}_SIGNATURE}ULL -I${KERNEL_DIR}/include
            --embed-dir=${PROGRAM_DIR}
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/thread_sum.c
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
            -lgcc -o ${SUM_ELF}
        DEPENDS
            thread_sum.c sum_binding.c sum.h ${PROGRAM_DIR}/sum.bin
            runtime-baremetal kernel
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${SUM}-baremetal-build DEPENDS ${SUM_ELF})
    add_dependencies(design-ctests-build ${SUM}-baremetal-build)
endforeach()

add_subdirectory(compiler)

# The RVV C tests of the thread's GP domain.
set(CTEST_PROFILE action)
add_ctest(rvv_integer_test.c)
