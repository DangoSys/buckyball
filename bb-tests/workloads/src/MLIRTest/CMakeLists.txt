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

set(FILECHECK ${LLVM_BUILD_DIR}/bin/FileCheck)

function(add_buckyball_mlir_contract TEST_NAME)
  cmake_parse_arguments(ARG "" "TARGET" "PASSES" ${ARGN})

  if(NOT DEFINED BUCKYBALL_MLIR_TEST_PREFIX)
    message(FATAL_ERROR "BUCKYBALL_MLIR_TEST_PREFIX is not set")
  endif()
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "${TEST_NAME}: TARGET is required")
  endif()
  list(FIND BUCKYBALL_TARGETS "${ARG_TARGET}" TARGET_INDEX)
  if(TARGET_INDEX EQUAL -1)
    message(FATAL_ERROR "${TEST_NAME}: unknown Buckyball target ${ARG_TARGET}")
  endif()

  if(NOT ARG_PASSES)
    set(ARG_PASSES ${BUCKYBALL_LOWER_BUCKYBALL})
  endif()

  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.mlir)
  set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}-${TEST_NAME}.out.mlir)
  set(TARGET ${BUCKYBALL_MLIR_TEST_PREFIX}-${ARG_TARGET}-${TEST_NAME}-mlir-test)

  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=${ARG_TARGET} ${ARG_PASSES} > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Buckyball MLIR contract IR: ${TARGET}"
    VERBATIM)

  add_custom_target(${TARGET}
    COMMAND ${FILECHECK} ${MLIR_SRC} --input-file=${OUT}
    DEPENDS ${OUT}
    COMMENT "Checking Buckyball MLIR contract: ${TARGET}"
    VERBATIM)

  if(DEFINED BUCKYBALL_MLIR_GROUP_TARGET)
    add_dependencies(${BUCKYBALL_MLIR_GROUP_TARGET} ${TARGET})
  endif()
endfunction()

# Bank-level / ball-op MLIR -> baremetal + linux ELF (same pipeline as toy MLIRTest).
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
  set(BB_BALL_ISA_INCLUDE_DIR "${BUCKYBALL_ISA_DIR}/${ARG_TARGET}")
  set(BB_BALL_ISA_H "${BB_BALL_ISA_INCLUDE_DIR}/ballISA.h")
  if(NOT EXISTS "${BB_BALL_ISA_H}")
    message(FATAL_ERROR "${NAME}: missing target ISA header ${BB_BALL_ISA_H}")
  endif()
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_main.cpp)
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}-${NAME}.o)
  set(PREFIX ${BUCKYBALL_MLIR_TEST_PREFIX})
  set(TEST_ID ${NAME})
  if(NOT NAME MATCHES "^${PREFIX}_")
    set(TEST_ID ${PREFIX}_${NAME})
  endif()
  set(BAREMETAL_BIN
    ${BUCKYBALL_WORKLOAD_CHIP}-${ARG_TARGET}-mlirtest-${TEST_ID}-baremetal)
  set(LINUX_BIN ${BUCKYBALL_WORKLOAD_CHIP}-${ARG_TARGET}-mlirtest-${TEST_ID}-linux)
  set(BAREMETAL_TARGET ${BAREMETAL_BIN}-build)
  set(LINUX_TARGET ${LINUX_BIN}-build)
  set(GROUP_TARGET ${BUCKYBALL_WORKLOAD_CHIP}-${ARG_TARGET}-mlirtest-${TEST_ID})

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
    set(MLIR_PASSES ${ARG_PASSES})
  else()
    set(MLIR_PASSES
      -assign-physical-banks
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
      ${BB_BALL_ISA_H}
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
          -lower-buckyball-intrinsics-to-rushb
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
    add_buckyball_rushb_native(${BUCKYBALL_WORKLOAD_CHIP}-${ARG_TARGET}-mlirtest-${TEST_ID}
      CXX
      OUTPUT_SUBDIR src/MLIRTest/rushB
      SOURCES ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C} ${MAIN_SRC} ${RUSHB_OBJ}
      INCLUDE_DIRS
        ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
        ${WORKLOAD_LIB_DIR}
        ${WORKLOAD_LIB_DIR}/bbhw/mem
        ${BB_BALL_ISA_INCLUDE_DIR}
      DEPENDS ${RUSHB_OBJ} ${MAIN_SRC} ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C}
              ${BB_BALL_ISA_H})
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
