# Cross-tile Linux control cores share one coherent LLC.
buckyball_ctest_name(POLY_CONTROL_NAME main chip-controllers)
set(POLY_CONTROL_ELF ${POLY_CONTROL_NAME}-multicore-baremetal)
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/controllers.c")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_CONTROL_ELF}
  COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${BBSIM_LD}
    -I${BUCKYBALL_ISA_DIR}/chip -I${WORKLOAD_LIB_DIR}/bbsw/baremetal
    -o ${POLY_CONTROL_ELF}
    ${CMAKE_CURRENT_SOURCE_DIR}/controllers_start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/controllers.c -lgcc
  DEPENDS controllers_start.S controllers.c ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S
    ${BBSIM_LD}
    ${BUCKYBALL_ISA_DIR}/chip/topology.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(${POLY_CONTROL_NAME}-ctest-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_CONTROL_ELF})
add_dependencies(poly-ctests-build ${POLY_CONTROL_NAME}-ctest-build)

set(BUCKYBALL_CTEST_TARGET main)
add_buckyball_linux_ctest(${POLY_CONTROL_NAME} ${CMAKE_CURRENT_SOURCE_DIR} controllers.c)
target_include_directories(${POLY_CONTROL_NAME}-linux PRIVATE ${BUCKYBALL_ISA_DIR}/chip)
target_link_libraries(${POLY_CONTROL_NAME}-linux PRIVATE pthread)
add_dependencies(poly-ctests-build ${POLY_CONTROL_NAME}-linux-build)

foreach(POLY_ANT_CASE mxmm shared bank bank_rows parallel highpa tlink tlink_bank tlink_reduce coherence)
set(POLY_ANT_CORE attention)
if(POLY_ANT_CASE STREQUAL "bank" OR POLY_ANT_CASE STREQUAL "bank_rows")
  set(POLY_ANT_CORE ffn)
endif()
buckyball_ctest_name(POLY_ANT_NAME ${POLY_ANT_CORE} chip-ant_${POLY_ANT_CASE})
foreach(_name BANK_WIDTH BANK_LINES BB_PRIVATE_VBANK_MAX)
  file(STRINGS "${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}/params.h" _line REGEX "^#define ${_name} ")
  string(REGEX REPLACE ".* " "" _${_name} "${_line}")
endforeach()
if(NOT _BANK_WIDTH EQUAL 128 OR _BANK_LINES LESS 128 OR _BB_PRIVATE_VBANK_MAX LESS 5)
  message(FATAL_ERROR
    "Ant Mxmm needs 128-bit rows, >=128 rows/bank and private banks through 5; actual width=${_BANK_WIDTH}, rows=${_BANK_LINES}, privateMax=${_BB_PRIVATE_VBANK_MAX}")
endif()
set(POLY_ANT_IMAGE ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_NAME}-ant)
set(ANT_RUNTIME_DIR ${BUCKYBALL_REPO_ROOT}/stack/runtime/src/ant)
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/ant_${POLY_ANT_CASE}.c")
add_custom_command(
  OUTPUT ${POLY_ANT_IMAGE}.elf ${POLY_ANT_IMAGE}.bin
  COMMAND ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
    -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
    -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${ANT_RUNTIME_DIR}/code.ld
    -I${WORKLOAD_LIB_DIR} -I${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}
    -I${BUCKYBALL_REPO_ROOT}/examples/balls/mxmm/workloads
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
    ${ANT_RUNTIME_DIR}/start.S ${CMAKE_CURRENT_SOURCE_DIR}/ant_${POLY_ANT_CASE}.c
    -o ${POLY_ANT_IMAGE}.elf
  COMMAND ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-objcopy
    -O binary --only-section=.text ${POLY_ANT_IMAGE}.elf ${POLY_ANT_IMAGE}.bin
  DEPENDS ant_${POLY_ANT_CASE}.c ant_${POLY_ANT_CASE}.h ${ANT_RUNTIME_DIR}/start.S ${ANT_RUNTIME_DIR}/code.ld
    ${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}/params.h ${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}/ballISA.h
    ${BUCKYBALL_BALL_ISA_DEPS})
add_custom_target(${POLY_ANT_NAME}-ant-build
  DEPENDS ${POLY_ANT_IMAGE}.elf ${POLY_ANT_IMAGE}.bin)
add_dependencies(poly-ctests-build ${POLY_ANT_NAME}-ant-build)

set(POLY_ANT_CONTROL_SOURCES)
if(POLY_ANT_CASE STREQUAL "bank" OR POLY_ANT_CASE STREQUAL "bank_rows" OR POLY_ANT_CASE STREQUAL "parallel")
  list(APPEND POLY_ANT_CONTROL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ant_${POLY_ANT_CASE}_exercise.c)
elseif(POLY_ANT_CASE STREQUAL "tlink_reduce")
  list(APPEND POLY_ANT_CONTROL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ant_tlink_reduce_control.c)
endif()
foreach(SOURCE IN LISTS POLY_ANT_CONTROL_SOURCES)
  buckyball_enforce_ctest_line_limit("${SOURCE}")
endforeach()

set(POLY_ANT_EMBED ${CMAKE_CURRENT_BINARY_DIR}/ant_${POLY_ANT_CASE}_image.S)
enable_language(ASM)
file(GENERATE OUTPUT ${POLY_ANT_EMBED} CONTENT
".section .rodata.ant,\"a\",@progbits\n.balign 8\n.global ant_image, ant_image_end\nant_image:\n.incbin \"${POLY_ANT_IMAGE}.bin\"\nant_image_end:\n.section .note.GNU-stack,\"\",@progbits\n")
set_source_files_properties(${POLY_ANT_EMBED} PROPERTIES
  OBJECT_DEPENDS "${POLY_ANT_IMAGE}.bin")
set(POLY_ANT_MAIN ant_${POLY_ANT_CASE}_main.c)
if(POLY_ANT_CASE STREQUAL "mxmm")
  set(POLY_ANT_MAIN ant_mxmm_main.cpp)
endif()
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}")
buckyball_ctest_name(POLY_ANT_HOST_NAME main chip-ant_${POLY_ANT_CASE})
set(BUCKYBALL_CTEST_TARGET ${POLY_ANT_CORE})
if(NOT POLY_ANT_CASE STREQUAL "highpa" AND NOT POLY_ANT_CASE MATCHES "^tlink" AND NOT POLY_ANT_CASE STREQUAL "coherence")
add_buckyball_linux_ctest(${POLY_ANT_HOST_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${POLY_ANT_MAIN})
target_sources(${POLY_ANT_HOST_NAME}-linux PRIVATE ${POLY_ANT_EMBED} ${POLY_ANT_CONTROL_SOURCES})
target_include_directories(${POLY_ANT_HOST_NAME}-linux PRIVATE ${BUCKYBALL_ISA_DIR}/chip)
target_link_libraries(${POLY_ANT_HOST_NAME}-linux PRIVATE runtime_ant pthread)
add_dependencies(${POLY_ANT_HOST_NAME}-linux ${POLY_ANT_NAME}-ant-build)
add_dependencies(poly-ctests-build ${POLY_ANT_HOST_NAME}-linux-build)

endif()

set(POLY_ANT_HOST_CORE ${POLY_ANT_CORE})
set(POLY_ANT_CONTROL_OBJECTS)
if(POLY_ANT_CASE STREQUAL "tlink_reduce")
  set(POLY_ANT_HOST_CORE main)
  set(POLY_ANT_CONTROL_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/ant_tlink_reduce_control.o)
  add_custom_command(OUTPUT ${POLY_ANT_CONTROL_OBJECTS}
    COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0 -fno-builtin
      -I${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}
      -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
      -c ${POLY_ANT_CONTROL_SOURCES} -o ${POLY_ANT_CONTROL_OBJECTS}
    DEPENDS ${POLY_ANT_CONTROL_SOURCES} ant_tlink_reduce.h
      ${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}/params.h)
  set(POLY_ANT_CONTROL_SOURCES)
endif()

set(POLY_ANT_HOST_ELF ${POLY_ANT_HOST_NAME}-multicore-baremetal)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_HOST_ELF}
  COMMAND ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${BBSIM_LD}
    -I${BUCKYBALL_ISA_DIR}/chip -I${BUCKYBALL_ISA_DIR}/${POLY_ANT_HOST_CORE}
    -I${WORKLOAD_LIB_DIR}
    -I${BUCKYBALL_REPO_ROOT}/examples/balls/f32add/workloads
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include -I${WORKLOAD_LIB_DIR}/bbsw/baremetal
    ${CMAKE_CURRENT_SOURCE_DIR}/controllers_start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN} ${POLY_ANT_CONTROL_SOURCES} ${POLY_ANT_CONTROL_OBJECTS} ${POLY_ANT_EMBED}
    ${BUCKYBALL_REPO_ROOT}/stack/runtime/src/ant.c
    ${BUCKYBALL_REPO_ROOT}/stack/runtime/src/tlink.c -lgcc -o ${POLY_ANT_HOST_ELF}
  DEPENDS ${POLY_ANT_MAIN} ant_${POLY_ANT_CASE}.h controllers_start.S ${POLY_ANT_EMBED}
    ${BBSW_BAREMETAL_DIR}/start.S ${BBSIM_LD}
    ${POLY_ANT_CONTROL_SOURCES} ${POLY_ANT_CONTROL_OBJECTS}
    ${BUCKYBALL_ISA_DIR}/${POLY_ANT_HOST_CORE}/params.h
    ${BUCKYBALL_ISA_DIR}/${POLY_ANT_HOST_CORE}/ballISA.h
    ${POLY_ANT_IMAGE}.bin ${BUCKYBALL_REPO_ROOT}/stack/runtime/src/ant.c
    ${BUCKYBALL_REPO_ROOT}/stack/runtime/include/ant.h
    ${BUCKYBALL_REPO_ROOT}/stack/runtime/src/tlink.c ${BUCKYBALL_REPO_ROOT}/stack/runtime/include/tlink.h
    ${BUCKYBALL_ISA_DIR}/${POLY_ANT_CORE}/params.h ${BUCKYBALL_ISA_DIR}/chip/topology.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(${POLY_ANT_HOST_NAME}-ctest-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_HOST_ELF})
add_dependencies(poly-ctests-build ${POLY_ANT_HOST_NAME}-ctest-build)
endforeach()

buckyball_ctest_name(POLY_STREAM_NAME main chip-ant_stream)
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/ant_stream_main.c")
add_executable(${POLY_STREAM_NAME}-linux ant_stream_main.c)
target_include_directories(${POLY_STREAM_NAME}-linux PRIVATE
  ${BUCKYBALL_ISA_DIR}/chip ${BUCKYBALL_ISA_DIR}/attention ${WORKLOAD_LIB_DIR}
  ${BUCKYBALL_REPO_ROOT}/examples/balls/mxmm/workloads)
target_link_libraries(${POLY_STREAM_NAME}-linux PRIVATE runtime_ant_tasks)
target_include_directories(runtime_ant_tasks PRIVATE ${BUCKYBALL_ISA_DIR}/attention)
add_dependencies(poly-ctests-build ${POLY_STREAM_NAME}-linux)
