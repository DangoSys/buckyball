# The mxint8quant C tests. Each test is a thread program (`<test>.c`, run on
# the thread's Ant through the Ball) and a host program (`<test>_main.c`, run
# on the core: it hands the operands to the thread and compares the result
# with vectors.h, generated offline by vectors.py from quant.py). Every core
# whose thread has the Ball runs every test.
set(BALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ENGINE_DIR ${BUCKYBALL_REPO_ROOT}/stack/engine)
foreach(HART IN LISTS DESIGN_DEVICE_CORES)
    set(THREADS ${DESIGN_DEVICE_CORES_${HART}_THREAD_INDICES})
    list(LENGTH THREADS THREAD_COUNT)
    if(THREAD_COUNT EQUAL 0)
        continue()
    endif()
    set(PROFILE ${DESIGN_THREADS_${THREADS}_PROFILE})
    if(NOT "mxint8quant" IN_LIST DESIGN_PROFILES_${PROFILE}_THREAD_BALLDOMAIN_BALLS)
        continue()
    endif()
    set(DEFINES ${DESIGN_TARGET_DEFS_${PROFILE}})
    list(TRANSFORM DEFINES PREPEND "-D")

    # quant_edges
    set(IMAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/quant_edges)
    if(NOT TARGET ${PROFILE}-mxint8quant-quant_edges-image)
        add_custom_command(
            OUTPUT ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            COMMAND ${CMAKE_COMMAND} -E make_directory ${IMAGE_DIR}
            COMMAND
                ${ELF_CC} -O2 -march=rv64im -mabi=lp64 -mcmodel=medany
                -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
                -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
                -nostdlib -nostartfiles -Wl,--no-relax
                -Wl,-T,${ENGINE_DIR}/program/code.ld ${DEFINES}
                "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${BALL_DIR}
                ${ENGINE_DIR}/program/start.S
                ${CMAKE_CURRENT_SOURCE_DIR}/quant_edges.c -o ${IMAGE_DIR}/image.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            DEPENDS
                quant_edges.c task.h vectors.h ${BALL_DIR}/isa/mxint8quant.h
                ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        add_custom_target(${PROFILE}-mxint8quant-quant_edges-image DEPENDS ${IMAGE_DIR}/image.bin)
    endif()
    set(TEST "${DESIGN_NAME}-core${HART}-ctest-${PROFILE}_mxint8quant_quant_edges")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREADS}_SIGNATURE}ULL
            -I${ENGINE_DIR}/include -I${BALL_DIR} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" --embed-dir=${IMAGE_DIR}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS> $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/quant_edges_main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
            -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        DEPENDS
            quant_edges_main.c binding.c task.h vectors.h ${IMAGE_DIR}/image.bin
            runtime-baremetal engine
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${TEST}-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
    add_dependencies(design-ctests-build ${TEST}-baremetal-build)

    # quant_random
    set(IMAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/quant_random)
    if(NOT TARGET ${PROFILE}-mxint8quant-quant_random-image)
        add_custom_command(
            OUTPUT ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            COMMAND ${CMAKE_COMMAND} -E make_directory ${IMAGE_DIR}
            COMMAND
                ${ELF_CC} -O2 -march=rv64im -mabi=lp64 -mcmodel=medany
                -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
                -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
                -nostdlib -nostartfiles -Wl,--no-relax
                -Wl,-T,${ENGINE_DIR}/program/code.ld ${DEFINES}
                "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${BALL_DIR}
                ${ENGINE_DIR}/program/start.S
                ${CMAKE_CURRENT_SOURCE_DIR}/quant_random.c -o ${IMAGE_DIR}/image.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            DEPENDS
                quant_random.c task.h vectors.h ${BALL_DIR}/isa/mxint8quant.h
                ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        add_custom_target(${PROFILE}-mxint8quant-quant_random-image DEPENDS ${IMAGE_DIR}/image.bin)
    endif()
    set(TEST "${DESIGN_NAME}-core${HART}-ctest-${PROFILE}_mxint8quant_quant_random")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREADS}_SIGNATURE}ULL
            -I${ENGINE_DIR}/include -I${BALL_DIR} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" --embed-dir=${IMAGE_DIR}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS> $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/quant_random_main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
            -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        DEPENDS
            quant_random_main.c binding.c task.h vectors.h ${IMAGE_DIR}/image.bin
            runtime-baremetal engine
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${TEST}-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
    add_dependencies(design-ctests-build ${TEST}-baremetal-build)
endforeach()
