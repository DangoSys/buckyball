# Embench: the benchmark `--bench embench`, the whole suite as one program
# (support/top.c runs every benchmark and checks its result): baremetal on the
# control hart, booted by the crt0 of the control hart of the C API, and a
# Linux program for the image of the bench. cubic is left
# out: its long double (__divtf3) overflows a relocation of the memory model.
set(CTEST_EMBENCH_WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EMBENCH_SUPPORT_DIR ${CTEST_EMBENCH_WORKLOAD_DIR}/support)
set(EMBENCH_SRC_DIR ${CTEST_EMBENCH_WORKLOAD_DIR}/src)

set(EMBENCH_TOP_SUPPORT_SOURCES
    ${EMBENCH_SUPPORT_DIR}/top.c
    ${EMBENCH_SUPPORT_DIR}/beebsc.c
    ${EMBENCH_SUPPORT_DIR}/boardsupport.c
    ${EMBENCH_SUPPORT_DIR}/chipsupport.c
)

set(EMBENCH_C_FLAGS
    -g
    -fno-common
    -O2
    -static
    -march=rv64gc
    -mcmodel=medany
    -fno-builtin-printf
    -specs=nano.specs
    -specs=nosys.specs
    -nostartfiles
    $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections
    -I${EMBENCH_SUPPORT_DIR}
)

set(EMBENCH_TOP_OBJ_FLAGS
    -g
    -fno-common
    -O2
    -march=rv64gc
    -mcmodel=medany
    -fno-builtin-printf
    -ffunction-sections
    -fdata-sections
    -I${EMBENCH_SUPPORT_DIR}
)

set(EMBENCH_LINUX_OBJ_FLAGS
    -g
    -fno-common
    -O2
    -march=rv64gc
    -fno-builtin-printf
    -ffunction-sections
    -fdata-sections
    -I${EMBENCH_SUPPORT_DIR}
)

macro(add_embench_top_object BMARK_NAME PREFIX SOURCE_FILES)
    set(OBJ "${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}.top.o")

    add_custom_command(
        OUTPUT ${OBJ}
        COMMAND
            ${ELF_CC} -r ${EMBENCH_TOP_OBJ_FLAGS}
            -Dbenchmark=${PREFIX}_benchmark
            -Dinitialise_benchmark=${PREFIX}_initialise_benchmark
            -Dwarm_caches=${PREFIX}_warm_caches
            -Dverify_benchmark=${PREFIX}_verify_benchmark -o ${OBJ}
            ${SOURCE_FILES}
        COMMAND
            ${ELF_OBJCOPY} --keep-global-symbol=${PREFIX}_benchmark
            --keep-global-symbol=${PREFIX}_initialise_benchmark
            --keep-global-symbol=${PREFIX}_warm_caches
            --keep-global-symbol=${PREFIX}_verify_benchmark ${OBJ}
        DEPENDS ${SOURCE_FILES}
        COMMENT "Building embench top object: ${BMARK_NAME}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
    )

    list(APPEND EMBENCH_TOP_OBJECTS ${OBJ})

    # The same benchmark for the Linux program, with the Linux compiler.
    set(LINUX_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}.linux.o")
    add_custom_command(
        OUTPUT ${LINUX_OBJ}
        COMMAND
            ${CMAKE_C_COMPILER} -r ${EMBENCH_LINUX_OBJ_FLAGS}
            -Dbenchmark=${PREFIX}_benchmark
            -Dinitialise_benchmark=${PREFIX}_initialise_benchmark
            -Dwarm_caches=${PREFIX}_warm_caches
            -Dverify_benchmark=${PREFIX}_verify_benchmark -o ${LINUX_OBJ}
            ${SOURCE_FILES}
        COMMAND
            ${ELF_OBJCOPY} --keep-global-symbol=${PREFIX}_benchmark
            --keep-global-symbol=${PREFIX}_initialise_benchmark
            --keep-global-symbol=${PREFIX}_warm_caches
            --keep-global-symbol=${PREFIX}_verify_benchmark ${LINUX_OBJ}
        DEPENDS ${SOURCE_FILES}
        COMMENT "Building embench Linux object: ${BMARK_NAME}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
    )

    list(APPEND EMBENCH_LINUX_OBJECTS ${LINUX_OBJ})
endmacro()

macro(add_embench_top_benchmark BMARK_NAME PREFIX)
    file(GLOB BMARK_SOURCES "${EMBENCH_SRC_DIR}/${BMARK_NAME}/*.c")
    add_embench_top_object(${BMARK_NAME} ${PREFIX} "${BMARK_SOURCES}")
endmacro()

function(add_embench_top_target)
    set(EMBENCH_TOP_OBJECTS)
    set(EMBENCH_LINUX_OBJECTS)

    add_embench_top_benchmark(aha-mont64 aha_mont64)
    add_embench_top_benchmark(crc32 crc32)
    add_embench_top_benchmark(edn edn)
    add_embench_top_benchmark(huffbench huffbench)
    add_embench_top_benchmark(matmult-int matmult_int)
    add_embench_top_benchmark(md5sum md5sum)
    add_embench_top_benchmark(minver minver)
    add_embench_top_benchmark(nbody nbody)
    add_embench_top_benchmark(nettle-aes nettle_aes)
    add_embench_top_benchmark(nettle-sha256 nettle_sha256)
    add_embench_top_benchmark(nsichneu nsichneu)
    add_embench_top_benchmark(picojpeg picojpeg)
    add_embench_top_benchmark(primecount primecount)
    add_embench_top_benchmark(qrduino qrduino)
    add_embench_top_benchmark(sglib-combined sglib_combined)
    add_embench_top_benchmark(slre slre)
    add_embench_top_benchmark(st st)
    add_embench_top_benchmark(statemate statemate)
    add_embench_top_benchmark(tarfind tarfind)
    add_embench_top_benchmark(ud ud)
    add_embench_top_benchmark(wikisort wikisort)

    # The whole suite as one program, the benchmark `--bench embench --os
    # baremetal` runs: <design>-embench-baremetal.
    set(EXECUTABLE "${DESIGN_NAME}-embench-baremetal")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMAND
            ${ELF_CC} ${EMBENCH_C_FLAGS} -o ${EXECUTABLE}
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${EMBENCH_TOP_SUPPORT_SOURCES} ${EMBENCH_TOP_OBJECTS} -lm
        DEPENDS
            ${EMBENCH_TOP_SUPPORT_SOURCES}
            ${EMBENCH_TOP_OBJECTS}
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMENT "Building embench top: ${EXECUTABLE}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
    )

    # The suite as a Linux program, embench: what the image `--bench embench`
    # (<design>-embench-linux) runs from its init.
    set(LINUX_PROGRAM ${CMAKE_CURRENT_BINARY_DIR}/linux/embench)
    add_custom_command(
        OUTPUT ${LINUX_PROGRAM}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/linux
        COMMAND
            ${CMAKE_C_COMPILER} -O2 -march=rv64gc -static -o ${LINUX_PROGRAM}
            ${EMBENCH_TOP_SUPPORT_SOURCES} ${EMBENCH_LINUX_OBJECTS} -lm
        DEPENDS ${EMBENCH_TOP_SUPPORT_SOURCES} ${EMBENCH_LINUX_OBJECTS}
        COMMENT "Building embench Linux program"
        VERBATIM
    )

    add_custom_target(
        embench
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE} ${LINUX_PROGRAM}
        COMMENT "Building embench for Buckyball"
    )
endfunction()

add_embench_top_target()
