# Embench: baremetal benchmarks sharing toy's linker script and startup
# code in stack/os/baremetal/.
set(CTEST_EMBENCH_WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EMBENCH_SUPPORT_DIR ${CTEST_EMBENCH_WORKLOAD_DIR}/support)
set(EMBENCH_SRC_DIR ${CTEST_EMBENCH_WORKLOAD_DIR}/src)
set(EMBENCH_THREAD ${BUCKYBALL_CONTROL_TARGET})

set(EMBENCH_SUPPORT_SOURCES
    ${EMBENCH_SUPPORT_DIR}/main.c
    ${EMBENCH_SUPPORT_DIR}/beebsc.c
    ${EMBENCH_SUPPORT_DIR}/boardsupport.c
    ${EMBENCH_SUPPORT_DIR}/chipsupport.c
)

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
    -Wl,-T,${BBSIM_LD}
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
    -I${WORKLOAD_LIB_DIR}
    -I${BUCKYBALL_ISA_DIR}/control
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
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
    -I${WORKLOAD_LIB_DIR}
    -I${BUCKYBALL_ISA_DIR}/control
    -ffunction-sections
    -fdata-sections
    -I${EMBENCH_SUPPORT_DIR}
)

function(add_embench_target BMARK_NAME)
    file(GLOB BMARK_SOURCES "${EMBENCH_SRC_DIR}/${BMARK_NAME}/*.c")
    set(EMBENCH_NAME
        "${BUCKYBALL_WORKLOAD_DESIGN}-${EMBENCH_THREAD}-ctest-benchmark-embench_${BMARK_NAME}"
    )
    set(EXECUTABLE "${EMBENCH_NAME}-baremetal")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMAND
            ${ELF_CC} ${EMBENCH_C_FLAGS} -o ${EXECUTABLE}
            ${BAREMETAL_DIR}/crt0.S ${BAREMETAL_DIR}/syscalls.c
            ${EMBENCH_SUPPORT_SOURCES} ${BMARK_SOURCES} -lm
        DEPENDS
            ${BMARK_SOURCES}
            ${EMBENCH_SUPPORT_SOURCES}
            ${BBSIM_LD}
            ${BAREMETAL_DIR}/crt0.S
            ${BAREMETAL_DIR}/syscalls.c
        COMMENT "Building embench: ${EXECUTABLE}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_custom_target(
        embench_${BMARK_NAME}
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMENT "Building embench ${BMARK_NAME}"
    )

    if(
        BMARK_NAME STREQUAL "matmult-int"
        OR BMARK_NAME STREQUAL "sglib-combined"
    )
        set(CHECK_NAME
            "${BUCKYBALL_WORKLOAD_DESIGN}-${EMBENCH_THREAD}-ctest-check-embench_${BMARK_NAME}"
        )
        set(CHECK_EXECUTABLE ${CHECK_NAME}-baremetal)
        set(CHECK_SUPPORT_SOURCES ${EMBENCH_SUPPORT_SOURCES})
        list(REMOVE_ITEM CHECK_SUPPORT_SOURCES ${EMBENCH_SUPPORT_DIR}/main.c)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CHECK_EXECUTABLE}
            COMMAND
                ${ELF_CC} ${EMBENCH_C_FLAGS} -o ${CHECK_EXECUTABLE}
                ${BAREMETAL_DIR}/crt0.S ${BAREMETAL_DIR}/syscalls.c
                ${EMBENCH_SUPPORT_DIR}/check.c ${CHECK_SUPPORT_SOURCES}
                ${BMARK_SOURCES} -lm
            DEPENDS
                ${EMBENCH_SUPPORT_DIR}/check.c
                ${CHECK_SUPPORT_SOURCES}
                ${BMARK_SOURCES}
                ${BAREMETAL_DIR}/crt0.S
                ${BAREMETAL_DIR}/syscalls.c
                ${BBSIM_LD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_custom_target(
            embench_check_${BMARK_NAME}
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CHECK_EXECUTABLE}
        )
        add_dependencies(embench_${BMARK_NAME} embench_check_${BMARK_NAME})
    endif()
endfunction()

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
    )

    list(APPEND EMBENCH_TOP_OBJECTS ${OBJ})
endmacro()

macro(add_embench_top_benchmark BMARK_NAME PREFIX)
    file(GLOB BMARK_SOURCES "${EMBENCH_SRC_DIR}/${BMARK_NAME}/*.c")
    add_embench_top_object(${BMARK_NAME} ${PREFIX} "${BMARK_SOURCES}")
endmacro()

function(add_embench_top_target)
    set(EMBENCH_TOP_OBJECTS)

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

    set(EMBENCH_TOP_NAME
        "${BUCKYBALL_WORKLOAD_DESIGN}-${EMBENCH_THREAD}-ctest-benchmark-embench_top"
    )
    set(EXECUTABLE "${EMBENCH_TOP_NAME}-baremetal")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMAND
            ${ELF_CC} ${EMBENCH_C_FLAGS} -o ${EXECUTABLE}
            ${BAREMETAL_DIR}/crt0.S ${BAREMETAL_DIR}/syscalls.c
            ${EMBENCH_TOP_SUPPORT_SOURCES} ${EMBENCH_TOP_OBJECTS} -lm
        DEPENDS
            ${EMBENCH_TOP_SUPPORT_SOURCES}
            ${BBSIM_LD}
            ${EMBENCH_TOP_OBJECTS}
            ${BAREMETAL_DIR}/crt0.S
            ${BAREMETAL_DIR}/syscalls.c
        COMMENT "Building embench top: ${EXECUTABLE}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_custom_target(
        embench_top
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMENT "Building embench top workload for Buckyball"
    )
endfunction()

#-------------------------------------------------------------------------------
# Add all embench benchmarks (19 benchmarks from embench-iot)
# Note: cubic uses long double (__divtf3) which causes RISC-V relocation
#       overflow with current memory model, excluded from embench-all
#-------------------------------------------------------------------------------
add_embench_target(aha-mont64)
add_embench_target(crc32)
add_embench_target(cubic)
add_embench_target(edn)
add_embench_target(huffbench)
add_embench_target(matmult-int)
add_embench_target(md5sum)
add_embench_target(minver)
add_embench_target(nbody)
add_embench_target(nettle-aes)
add_embench_target(nettle-sha256)
add_embench_target(nsichneu)
add_embench_target(picojpeg)
add_embench_target(primecount)
add_embench_target(qrduino)
add_embench_target(sglib-combined)
add_embench_target(slre)
add_embench_target(st)
add_embench_target(statemate)
add_embench_target(tarfind)
add_embench_target(ud)
add_embench_target(wikisort)
add_embench_top_target()

# Master embench build target (excludes cubic due to long double link issue)
add_custom_target(
    embench-all
    DEPENDS
        embench_aha-mont64
        embench_crc32
        embench_edn
        embench_huffbench
        embench_matmult-int
        embench_md5sum
        embench_minver
        embench_nbody
        embench_nettle-aes
        embench_nettle-sha256
        embench_nsichneu
        embench_picojpeg
        embench_primecount
        embench_qrduino
        embench_sglib-combined
        embench_slre
        embench_st
        embench_statemate
        embench_tarfind
        embench_ud
        embench_wikisort
        embench_top
    COMMENT
        "Building all embench benchmarks for Buckyball (21/22, cubic excluded)"
)
