# The workload root chooses the compiler instance for this chip's core.
if(NOT DEFINED BUDDY_BINARY_DIR)
  message(FATAL_ERROR
    "BUDDY_BINARY_DIR must be set by the workload root.")
endif()

if(NOT DEFINED LLVM_BUILD_DIR)
  message(FATAL_ERROR "LLVM_BUILD_DIR is not set")
endif()

if(NOT DEFINED BUDDY_OPT)
  message(FATAL_ERROR "BUDDY_OPT is not set")
endif()

# Bank-level / ball-op MLIR -> baremetal + linux ELF (same pipeline as toy MLIRTest).
function(buckyball_enforce_mlir_test_line_limit SOURCE_FILE)
  if(NOT EXISTS "${SOURCE_FILE}")
    message(FATAL_ERROR "MLIR test source does not exist: ${SOURCE_FILE}")
  endif()
  file(READ "${SOURCE_FILE}" _content)
  string(REGEX MATCHALL "\n" _newlines "${_content}")
  list(LENGTH _newlines _lines)
  string(REGEX MATCH "[^\n]$" _trailing_content "${_content}")
  if(_trailing_content)
    math(EXPR _lines "${_lines} + 1")
  endif()
  if(_lines GREATER 100)
    message(FATAL_ERROR
      "MLIR test exceeds 100 lines: ${SOURCE_FILE} (${_lines} lines). "
      "Split it into one focused test per file.")
  endif()
endfunction()

function(buckyball_require_smatmul_bank_shape TARGET)
  set(_params "${BUCKYBALL_ISA_DIR}/${TARGET}/params.h")
  file(STRINGS "${_params}" _bank_lines REGEX "^#define BANK_LINES ")
  file(STRINGS "${_params}" _bank_width REGEX "^#define BANK_WIDTH ")
  string(REGEX REPLACE ".* " "" _lines "${_bank_lines}")
  string(REGEX REPLACE ".* " "" _width "${_bank_width}")
  math(EXPR _line_remainder "${_lines} % 64")
  if(NOT (_width EQUAL 128) OR _lines LESS 64 OR
     NOT (_line_remainder EQUAL 0))
    message(FATAL_ERROR
      "SMatMul BankOp test requires BANK_WIDTH=128 and BANK_LINES a multiple "
      "of 64 (got width=${_width}, lines=${_lines})")
  endif()
endfunction()

function(buckyball_require_quant_i32_bank_shape TARGET)
  set(_params "${BUCKYBALL_ISA_DIR}/${TARGET}/params.h")
  file(STRINGS "${_params}" _bank_lines REGEX "^#define BANK_LINES ")
  string(REGEX REPLACE ".* " "" _lines "${_bank_lines}")
  math(EXPR _remainder "${_lines} % 4")
  if(_lines LESS 4 OR NOT (_remainder EQUAL 0))
    message(FATAL_ERROR
      "ToInt8 INT32 BankOp test requires BANK_LINES divisible by four "
      "(got lines=${_lines})")
  endif()
endfunction()

function(buckyball_require_mxfp2int_bank_shape TARGET)
  set(_params "${BUCKYBALL_ISA_DIR}/${TARGET}/params.h")
  file(STRINGS "${_params}" _bank_lines REGEX "^#define BANK_LINES ")
  string(REGEX REPLACE ".* " "" _lines "${_bank_lines}")
  math(EXPR _remainder "${_lines} % 32")
  if(_lines LESS 32 OR NOT (_remainder EQUAL 0))
    message(FATAL_ERROR
      "Mxfp2Int BankOp test requires BANK_LINES divisible by 32 "
      "(got lines=${_lines})")
  endif()
endfunction()

function(buckyball_require_min_bank_lines TARGET MINIMUM BALL)
  set(_params "${BUCKYBALL_ISA_DIR}/${TARGET}/params.h")
  file(STRINGS "${_params}" _bank_lines REGEX "^#define BANK_LINES ")
  string(REGEX REPLACE ".* " "" _lines "${_bank_lines}")
  if(_lines LESS MINIMUM)
    message(FATAL_ERROR
      "${BALL} BankOp test requires BANK_LINES >= ${MINIMUM} (got ${_lines})")
  endif()
endfunction()

function(add_buckyball_mlir_test NAME)
  cmake_parse_arguments(ARG "" "TARGET" "PASSES" ${ARGN})

  if(NOT DEFINED BUCKYBALL_MLIR_TEST_PREFIX)
    message(FATAL_ERROR "BUCKYBALL_MLIR_TEST_PREFIX is not set")
  endif()
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before mlir tests")
  endif()
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "${NAME}: TARGET is required")
  endif()
  list(FIND BUCKYBALL_TARGETS "${ARG_TARGET}" TARGET_INDEX)
  if(TARGET_INDEX EQUAL -1)
    message(FATAL_ERROR "${NAME}: unknown Buckyball target ${ARG_TARGET}")
  endif()
  buckyball_assign_physical_banks(${ARG_TARGET} ASSIGN_PHYSICAL_BANKS)
  set(BB_BALL_ISA_INCLUDE_DIR "${BUCKYBALL_ISA_DIR}/${ARG_TARGET}")
  set(BB_BALL_ISA_H "${BB_BALL_ISA_INCLUDE_DIR}/ballISA.h")
  set(BB_PARAMS_H "${BB_BALL_ISA_INCLUDE_DIR}/params.h")
  if(NOT EXISTS "${BB_BALL_ISA_H}")
    message(FATAL_ERROR "${NAME}: missing target ISA header ${BB_BALL_ISA_H}")
  endif()
  if(NOT EXISTS "${BB_PARAMS_H}")
    message(FATAL_ERROR "${NAME}: missing target params.h ${BB_PARAMS_H}")
  endif()
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.mlir)
  set(MLIR_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.mlir.in)
  if(EXISTS ${MLIR_TEMPLATE})
    file(STRINGS ${BB_PARAMS_H} _bank_lines REGEX "^#define BANK_LINES ")
    file(STRINGS ${BB_PARAMS_H} _bank_width REGEX "^#define BANK_WIDTH ")
    file(STRINGS ${BB_PARAMS_H} _bank_isqrt REGEX "^#define BANK_ISQRT ")
    string(REGEX REPLACE ".* " "" BB_BANK_LINES "${_bank_lines}")
    string(REGEX REPLACE ".* " "" BB_BANK_WIDTH_BITS "${_bank_width}")
    string(REGEX REPLACE ".* " "" BB_BANK_ISQRT "${_bank_isqrt}")
    math(EXPR BB_BANK_I8_LANES "${BB_BANK_WIDTH_BITS} / 8")
    math(EXPR BB_BANK_I32_LANES "${BB_BANK_WIDTH_BITS} / 32")
    math(EXPR BB_BANK_SQUARE "${BB_BANK_ISQRT} * ${BB_BANK_ISQRT}")
    math(EXPR BB_SMM_M "${BB_BANK_LINES} / 4")
    math(EXPR BB_SMM_CONFIG "${BB_SMM_M} | (16 << 12) | (16 << 24)")
    math(EXPR BB_QUANT_I32_OUTPUT_ROWS "${BB_BANK_LINES} / 4")
    math(EXPR BB_MXFP_BLOCKS "${BB_BANK_LINES} / 2")
    math(EXPR BB_MXFP_MMIO_ROWS "${BB_MXFP_BLOCKS} / 16")
    if(BB_BANK_LINES LESS 1 OR BB_BANK_ISQRT LESS 1 OR BB_BANK_I8_LANES LESS 1 OR
       BB_BANK_I32_LANES LESS 1)
      message(FATAL_ERROR "${NAME}: invalid generated bank parameters")
    endif()
    set(MLIR_SRC ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}-${NAME}.mlir)
    configure_file(${MLIR_TEMPLATE} ${MLIR_SRC} @ONLY)
  endif()
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_main.cpp)
  buckyball_enforce_mlir_test_line_limit(${MLIR_SRC})
  buckyball_enforce_mlir_test_line_limit(${MAIN_SRC})
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}-${NAME}.o)
  set(PREFIX ${BUCKYBALL_MLIR_TEST_PREFIX})
  set(TEST_CASE ${NAME})
  if(NAME MATCHES "^${PREFIX}_(.+)$")
    set(TEST_CASE ${CMAKE_MATCH_1})
  endif()
  buckyball_mlirtest_name(TEST_NAME ${ARG_TARGET} ${PREFIX}_${TEST_CASE})
  set(BAREMETAL_BIN ${TEST_NAME}-baremetal)
  set(LINUX_BIN ${TEST_NAME}-linux)
  set(BAREMETAL_TARGET ${BAREMETAL_BIN}-build)
  set(LINUX_TARGET ${LINUX_BIN}-build)
  set(GROUP_TARGET ${TEST_NAME})

  set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
  set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -DBAREMETAL -Wl,-T,${BBSIM_LD})
  set(LINUX_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
  set(LINUX_FLAGS -static -Wl,--no-dynamic-linker -march=rv64gc)
  set(CRUNNER_UTILS_SRC ${WORKLOAD_LIB_DIR}/bbsw/CRunnerUtils/CRunnerUtils.cpp)
  set(BBHW_MEM_C ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)
  set(LLVM_MLIR_EXECUTION_ENGINE_DIR
    ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)

  if(ARG_PASSES)
    set(MLIR_PASSES)
    foreach(PASS IN LISTS ARG_PASSES)
      if(PASS STREQUAL "-assign-physical-banks")
        list(APPEND MLIR_PASSES ${ASSIGN_PHYSICAL_BANKS})
      else()
        list(APPEND MLIR_PASSES ${PASS})
      endif()
    endforeach()
  else()
    set(MLIR_PASSES
      ${ASSIGN_PHYSICAL_BANKS}
      ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
      -convert-linalg-to-loops
      -expand-strided-metadata
      -lower-affine
      -convert-scf-to-cf
      -convert-cf-to-llvm
      ${BUCKYBALL_LOWER_BUCKYBALL}
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts)
  endif()
  string(JOIN " " MLIR_PASS_COMMAND ${MLIR_PASSES})

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND bash -o pipefail -c
      "${BUDDY_OPT} ${MLIR_SRC} --target=${ARG_TARGET} ${MLIR_PASS_COMMAND} | ${BUDDY_TRANSLATE} --buddy-to-llvmir | ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o ${OBJ}"
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building ${NAME}.o from ${PREFIX} MLIR"
    VERBATIM)

  add_custom_command(
    OUTPUT ${BAREMETAL_BIN}
    COMMAND ${ELF_CC} ${C_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -I${WORKLOAD_LIB_DIR}
      -I${WORKLOAD_LIB_DIR}/bbhw/mem
      -I${BB_BALL_ISA_INCLUDE_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ} ${MAIN_SRC} ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C}
      ${BB_BALL_ISA_H} ${BB_PARAMS_H}
      ${BBSW_BAREMETAL_DIR}/crt0.S ${BBSW_BAREMETAL_DIR}/syscalls.c
    COMMENT "Linking ${BAREMETAL_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

  add_custom_command(
    OUTPUT ${LINUX_BIN}
    COMMAND ${LINUX_CXX} ${LINUX_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -I${WORKLOAD_LIB_DIR}
      -I${WORKLOAD_LIB_DIR}/bbhw/mem
      -I${BB_BALL_ISA_INCLUDE_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ} ${MAIN_SRC} ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C} ${BB_BALL_ISA_H}
            ${BB_PARAMS_H}
    COMMENT "Linking ${LINUX_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

  add_custom_target(${BAREMETAL_TARGET}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})
  add_custom_target(${LINUX_TARGET}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})
  add_custom_target(${GROUP_TARGET}
    DEPENDS ${BAREMETAL_TARGET} ${LINUX_TARGET})

  if(DEFINED BUCKYBALL_RUSHB_BEMU_MANIFEST AND
     DEFINED BUCKYBALL_RUSHB_VERILATOR_LIBRARY)
    if(NOT DEFINED BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS OR
       NOT DEFINED BUCKYBALL_LOWER_BUCKYBALL_RUSHB)
      message(FATAL_ERROR
        "rushB MLIR lowers require BUCKYBALL_LOWER_*_RUSHB "
        "(define them in bb-tests/workloads/CMakeLists.txt)")
    endif()
    buckyball_rushb_default_core(_rushb_core "${ARG_TARGET}")
    set(RUSHB_MLIR_PASSES)
    set(RUSHB_HAS_RECONCILE FALSE)
    foreach(MLIR_PASS ${MLIR_PASSES})
      if(MLIR_PASS STREQUAL "${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}")
        list(APPEND RUSHB_MLIR_PASSES ${BUCKYBALL_LOWER_BANK_SSA_TO_RUSHB_INTRINSICS})
      elseif(MLIR_PASS STREQUAL "${BUCKYBALL_LOWER_BUCKYBALL}")
        list(APPEND RUSHB_MLIR_PASSES ${BUCKYBALL_LOWER_BUCKYBALL_RUSHB})
      elseif(MLIR_PASS STREQUAL "-reconcile-unrealized-casts")
        set(RUSHB_HAS_RECONCILE TRUE)
        list(APPEND RUSHB_MLIR_PASSES
          "-lower-buckyball-intrinsics-to-rushb=core_id=${_rushb_core}"
          -reconcile-unrealized-casts)
      else()
        list(APPEND RUSHB_MLIR_PASSES ${MLIR_PASS})
      endif()
    endforeach()
    if(NOT RUSHB_HAS_RECONCILE)
      message(FATAL_ERROR
        "rushB pass rewrite for ${NAME} requires -reconcile-unrealized-casts")
    endif()
    set(RUSHB_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${NAME}-rushB.o)
    string(JOIN " " RUSHB_MLIR_PASS_COMMAND ${RUSHB_MLIR_PASSES})
    add_custom_command(
      OUTPUT ${RUSHB_OBJ}
      COMMAND bash -o pipefail -c
        "${BUDDY_OPT} ${MLIR_SRC} --target=${ARG_TARGET} ${RUSHB_MLIR_PASS_COMMAND} | ${BUDDY_TRANSLATE} --buddy-to-llvmir | ${BUDDY_LLC} -filetype=obj -mtriple=x86_64 -O2 -o ${RUSHB_OBJ}"
      DEPENDS ${MLIR_SRC} ${BUDDY_OPT} ${BUDDY_TRANSLATE} ${BUDDY_LLC}
      COMMENT "Building rushB ${NAME}.o from ${PREFIX} MLIR"
      VERBATIM)
    add_buckyball_rushb_native(${TEST_NAME}
      CXX
      OUTPUT_SUBDIR src/MLIRTest/rushB
      CORE_ID ${_rushb_core}
      SOURCES ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C} ${MAIN_SRC} ${RUSHB_OBJ}
      INCLUDE_DIRS
        ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
        ${WORKLOAD_LIB_DIR}
        ${WORKLOAD_LIB_DIR}/bbhw/mem
        ${BB_BALL_ISA_INCLUDE_DIR}
      DEPENDS ${RUSHB_OBJ} ${MAIN_SRC} ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C}
              ${BB_BALL_ISA_H} ${BB_PARAMS_H})
  endif()

  if(DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
    add_dependencies(${BUCKYBALL_MLIR_GROUP_TARGET} ${GROUP_TARGET})
  endif()
endfunction()

set(MLIRTEST_BASE_DIR ${MLIRTEST_WORKLOAD_DIR}/base)

add_subdirectory(${MLIRTEST_WORKLOAD_DIR}/base
  ${CMAKE_CURRENT_BINARY_DIR}/mlir_tests/base)

add_custom_target(balls-mlir-tests-build)
buckyball_add_ball_workload_subdirs(mlir_tests)

add_custom_target(MLIRTest-all ALL DEPENDS
  base-mlirtest-build
  balls-mlir-tests-build
)
