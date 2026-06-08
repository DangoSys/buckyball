set(ELF_CC "riscv64-unknown-elf-gcc")
set(ELF_OBJCOPY "riscv64-unknown-elf-objcopy")

#-------------------------------------------------------------------------------
# Embench: baremetal benchmarks (singlecore + multicore) sharing toy's linker
# script and startup code in lib/bbsw/baremetal/.
#-------------------------------------------------------------------------------
set(CTEST_EMBENCH_WORKLOAD_DIR ${CTEST_WORKLOAD_DIR}/embench)
set(EMBENCH_SUPPORT_DIR ${CTEST_EMBENCH_WORKLOAD_DIR}/support)
set(EMBENCH_SRC_DIR ${CTEST_EMBENCH_WORKLOAD_DIR}/src)
set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)

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
  -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD}
  -I${WORKLOAD_LIB_DIR}
  -ffunction-sections -fdata-sections
  -Wl,--gc-sections
  -I${EMBENCH_SUPPORT_DIR}
)

set(EMBENCH_TOP_OBJ_FLAGS
  -g -fno-common -O2 -march=rv64gc -mcmodel=medany
  -fno-builtin-printf
  -I${WORKLOAD_LIB_DIR}
  -ffunction-sections -fdata-sections
  -I${EMBENCH_SUPPORT_DIR}
)

# Generate single-core baremetal embench target
function(add_embench_singlecore_target BMARK_NAME SOURCE_FILES)
  set(EXECUTABLE "embench_${BMARK_NAME}_singlecore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${EMBENCH_C_FLAGS} -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${EMBENCH_SUPPORT_SOURCES}
      ${SOURCE_FILES}
      -lm
    DEPENDS ${SOURCE_FILES} ${EMBENCH_SUPPORT_SOURCES}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
    COMMENT "Building embench singlecore: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(embench_${BMARK_NAME}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

# Generate multi-core baremetal embench target
function(add_embench_multicore_target BMARK_NAME SOURCE_FILES)
  set(EXECUTABLE "embench_${BMARK_NAME}_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${EMBENCH_C_FLAGS} -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/start.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      -DMULTICORE=3
      ${EMBENCH_SUPPORT_SOURCES}
      ${SOURCE_FILES}
      -lm
    DEPENDS ${SOURCE_FILES} ${EMBENCH_SUPPORT_SOURCES}
      ${BBSW_BAREMETAL_DIR}/start.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
    COMMENT "Building embench multicore: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(embench_${BMARK_NAME}_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

# Cross-platform embench target (singlecore + multicore baremetal)
function(add_embench_target BMARK_NAME)
  # Collect all .c files in this benchmark's directory
  file(GLOB BMARK_SOURCES "${EMBENCH_SRC_DIR}/${BMARK_NAME}/*.c")

  add_embench_singlecore_target(${BMARK_NAME} "${BMARK_SOURCES}")
  add_embench_multicore_target(${BMARK_NAME} "${BMARK_SOURCES}")

  add_custom_target(embench_${BMARK_NAME}
    DEPENDS embench_${BMARK_NAME}_singlecore_baremetal
            embench_${BMARK_NAME}_multicore_baremetal
    COMMENT "Building embench ${BMARK_NAME} for all platforms"
  )
endfunction()

macro(add_embench_top_object BMARK_NAME PREFIX SOURCE_FILES)
  set(OBJ "${CMAKE_CURRENT_BINARY_DIR}/${PREFIX}.top.o")

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND ${ELF_CC} -r ${EMBENCH_TOP_OBJ_FLAGS}
      -Dbenchmark=${PREFIX}_benchmark
      -Dinitialise_benchmark=${PREFIX}_initialise_benchmark
      -Dwarm_caches=${PREFIX}_warm_caches
      -Dverify_benchmark=${PREFIX}_verify_benchmark
      -o ${OBJ}
      ${SOURCE_FILES}
    COMMAND ${ELF_OBJCOPY}
      --keep-global-symbol=${PREFIX}_benchmark
      --keep-global-symbol=${PREFIX}_initialise_benchmark
      --keep-global-symbol=${PREFIX}_warm_caches
      --keep-global-symbol=${PREFIX}_verify_benchmark
      ${OBJ}
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
  add_embench_top_benchmark(minver minver)
  add_embench_top_benchmark(nbody nbody)
  add_embench_top_benchmark(nettle-aes nettle_aes)
  add_embench_top_benchmark(nettle-sha256 nettle_sha256)
  add_embench_top_benchmark(nsichneu nsichneu)
  add_embench_top_benchmark(picojpeg picojpeg)
  add_embench_top_benchmark(qrduino qrduino)
  add_embench_top_benchmark(sglib-combined sglib_combined)
  add_embench_top_benchmark(slre slre)
  add_embench_top_benchmark(st st)
  add_embench_top_benchmark(statemate statemate)
  add_embench_top_benchmark(ud ud)
  add_embench_top_benchmark(wikisort wikisort)

  foreach(MODE singlecore multicore)
    set(EXECUTABLE "embench_top_${MODE}-baremetal")
    if(MODE STREQUAL "singlecore")
      set(STARTUP ${BBSW_BAREMETAL_DIR}/crt0.S)
      set(MODE_DEFS)
    else()
      set(STARTUP ${BBSW_BAREMETAL_DIR}/start.S)
      set(MODE_DEFS -DMULTICORE=3)
    endif()

    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
      COMMAND ${ELF_CC} ${EMBENCH_C_FLAGS} ${MODE_DEFS} -o ${EXECUTABLE}
        ${STARTUP}
        ${BBSW_BAREMETAL_DIR}/syscalls.c
        ${EMBENCH_TOP_SUPPORT_SOURCES}
        ${EMBENCH_TOP_OBJECTS}
        ${BBSW_BAREMETAL_DIR}/test.c
        -lm
      DEPENDS ${EMBENCH_TOP_SUPPORT_SOURCES}
        ${EMBENCH_TOP_OBJECTS}
        ${STARTUP}
        ${BBSW_BAREMETAL_DIR}/syscalls.c
        ${BBSW_BAREMETAL_DIR}/test.c
      COMMENT "Building embench top ${MODE}: ${EXECUTABLE}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_custom_target(embench_top_${MODE}_baremetal
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
  endforeach()

  add_custom_target(embench_top DEPENDS
    embench_top_singlecore_baremetal
    embench_top_multicore_baremetal
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
add_embench_target(minver)
add_embench_target(nbody)
add_embench_target(nettle-aes)
add_embench_target(nettle-sha256)
add_embench_target(nsichneu)
add_embench_target(picojpeg)
add_embench_target(qrduino)
add_embench_target(sglib-combined)
add_embench_target(slre)
add_embench_target(st)
add_embench_target(statemate)
add_embench_target(ud)
add_embench_target(wikisort)
add_embench_top_target()

# Master embench build target (excludes cubic due to long double link issue)
add_custom_target(embench-all DEPENDS
  embench_aha-mont64 embench_crc32 embench_edn
  embench_huffbench embench_matmult-int embench_minver embench_nbody
  embench_nettle-aes embench_nettle-sha256 embench_nsichneu
  embench_picojpeg embench_qrduino embench_sglib-combined
  embench_slre embench_st embench_statemate embench_ud embench_wikisort
  embench_top
  COMMENT "Building all embench benchmarks for Buckyball (18/19, cubic excluded)"
)
