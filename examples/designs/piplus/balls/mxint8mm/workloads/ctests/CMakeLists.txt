# The mxint8mm C tests. Each test is a thread program (`<test>.c`, run on
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
    if(NOT "mxint8mm" IN_LIST DESIGN_PROFILES_${PROFILE}_THREAD_BALLDOMAIN_BALLS)
        continue()
    endif()
    set(DEFINES ${DESIGN_TARGET_DEFS_${PROFILE}})
    list(TRANSFORM DEFINES PREPEND "-D")

    # mm_small
    set(IMAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/mm_small)
    if(NOT TARGET ${PROFILE}-mxint8mm-mm_small-image)
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
                ${CMAKE_CURRENT_SOURCE_DIR}/mm_small.c -o ${IMAGE_DIR}/image.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            DEPENDS
                mm_small.c task.h vectors.h ${BALL_DIR}/isa/mxint8mm.h
                ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        add_custom_target(${PROFILE}-mxint8mm-mm_small-image DEPENDS ${IMAGE_DIR}/image.bin)
    endif()
    set(TEST "${DESIGN_NAME}-core${HART}-ctest-${PROFILE}_mxint8mm_mm_small")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREADS}_SIGNATURE}ULL
            -I${ENGINE_DIR}/include -I${BALL_DIR} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" --embed-dir=${IMAGE_DIR}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS> $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/mm_small_main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
            -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        DEPENDS
            mm_small_main.c binding.c task.h vectors.h ${IMAGE_DIR}/image.bin
            runtime-baremetal engine
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${TEST}-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
    add_dependencies(design-ctests-build ${TEST}-baremetal-build)

    # mm_accumulate
    set(IMAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/mm_accumulate)
    if(NOT TARGET ${PROFILE}-mxint8mm-mm_accumulate-image)
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
                ${CMAKE_CURRENT_SOURCE_DIR}/mm_accumulate.c -o ${IMAGE_DIR}/image.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            DEPENDS
                mm_accumulate.c task.h vectors.h ${BALL_DIR}/isa/mxint8mm.h
                ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        add_custom_target(${PROFILE}-mxint8mm-mm_accumulate-image DEPENDS ${IMAGE_DIR}/image.bin)
    endif()
    set(TEST "${DESIGN_NAME}-core${HART}-ctest-${PROFILE}_mxint8mm_mm_accumulate")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREADS}_SIGNATURE}ULL
            -I${ENGINE_DIR}/include -I${BALL_DIR} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" --embed-dir=${IMAGE_DIR}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS> $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/mm_accumulate_main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
            -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        DEPENDS
            mm_accumulate_main.c binding.c task.h vectors.h ${IMAGE_DIR}/image.bin
            runtime-baremetal engine
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${TEST}-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
    add_dependencies(design-ctests-build ${TEST}-baremetal-build)

    # mm_extremes
    set(IMAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/mm_extremes)
    if(NOT TARGET ${PROFILE}-mxint8mm-mm_extremes-image)
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
                ${CMAKE_CURRENT_SOURCE_DIR}/mm_extremes.c -o ${IMAGE_DIR}/image.elf
            COMMAND
                ${ELF_OBJCOPY} -O binary --only-section=.text
                ${IMAGE_DIR}/image.elf ${IMAGE_DIR}/image.bin
            DEPENDS
                mm_extremes.c task.h vectors.h ${BALL_DIR}/isa/mxint8mm.h
                ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
        add_custom_target(${PROFILE}-mxint8mm-mm_extremes-image DEPENDS ${IMAGE_DIR}/image.bin)
    endif()
    set(TEST "${DESIGN_NAME}-core${HART}-ctest-${PROFILE}_mxint8mm_mm_extremes")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        COMMAND
            ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
            -DTHREAD_SIGNATURE=${DESIGN_THREADS_${THREADS}_SIGNATURE}ULL
            -I${ENGINE_DIR}/include -I${BALL_DIR} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${PROFILE}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" --embed-dir=${IMAGE_DIR}
            $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS> $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/mm_extremes_main.c
            ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
            $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
            -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
        DEPENDS
            mm_extremes_main.c binding.c task.h vectors.h ${IMAGE_DIR}/image.bin
            runtime-baremetal engine
            $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${TEST}-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
    add_dependencies(design-ctests-build ${TEST}-baremetal-build)
endforeach()
