# ===============================================================
# SoC tests: CPU-visible chip devices (CLINT, PLIC, SCU) exercised on
# the chip's boot hart, independent of any Ball ISA.
# ===============================================================
set(SOCTEST_MAX_LINES 100)
set(BUCKYBALL_SOCTEST_C_FLAGS
  -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD}
)

# The target whose generated profile boots hart 0 owns the SoC tests.
unset(BUCKYBALL_SOCTEST_TARGET)
foreach(_target IN LISTS BUCKYBALL_TARGETS)
  file(STRINGS "${BUCKYBALL_ISA_DIR}/${_target}/params.h" _boot
    REGEX "^#define BB_TEST_HART 0$")
  if(_boot)
    set(BUCKYBALL_SOCTEST_TARGET ${_target})
    break()
  endif()
endforeach()
if(NOT BUCKYBALL_SOCTEST_TARGET)
  message(FATAL_ERROR
    "SoC tests need a target with BB_TEST_HART 0; targets: ${BUCKYBALL_TARGETS}")
endif()

add_custom_target(soctests-build)

function(add_buckyball_soctest SOURCE_FILE)
  get_filename_component(_stem ${SOURCE_FILE} NAME_WE)
  buckyball_soctest_name(TEST_NAME ${BUCKYBALL_SOCTEST_TARGET} ${_stem})
  set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
  file(STRINGS ${SOURCE} _lines)
  list(LENGTH _lines _n)
  if(_n GREATER SOCTEST_MAX_LINES)
    message(FATAL_ERROR
      "soctest exceeds ${SOCTEST_MAX_LINES} lines: ${SOURCE} (${_n} lines)")
  endif()

  set(EXECUTABLE "${TEST_NAME}-baremetal")
  set(PARAMS ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_SOCTEST_TARGET}/params.h)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${BUCKYBALL_SOCTEST_C_FLAGS}
      -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${SOURCE}
      -I${BUCKYBALL_ISA_DIR}/chip
      -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_SOCTEST_TARGET}
      -I${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${SOURCE} ${PARAMS} ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c ${CMAKE_CURRENT_SOURCE_DIR}/soc.h
    COMMENT "Building SoC test: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_custom_target(${TEST_NAME}-baremetal-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
  add_dependencies(soctests-build ${TEST_NAME}-baremetal-build)
endfunction()

add_buckyball_soctest(clint_timer_interrupt.c)
add_buckyball_soctest(clint_software_interrupt.c)
add_buckyball_soctest(clint_supervisor_timer.c)
add_buckyball_soctest(time_csr_clock.c)
add_buckyball_soctest(time_csr_permissions.c)
add_buckyball_soctest(lrsc_word.c)
add_buckyball_soctest(lrsc_translated.c)
add_buckyball_soctest(plic_register_access.c)

add_custom_target(sync-soctest-bin
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/SocTest
  COMMAND rsync -a --delete --exclude=CMakeFiles "--include=*/"
          "--include=*-baremetal" "--include=*-baremetal.expect" "--exclude=*"
          ${CMAKE_CURRENT_BINARY_DIR}/ ${OUTPUT_BIN_DIR}/src/SocTest/
  DEPENDS soctests-build
  COMMENT "Syncing SoC test binaries to output directory")
