set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -DBAREMETAL -Wl,-T,${BBSIM_LD})
set(LINUX_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
set(LINUX_FLAGS -static -Wl,--no-dynamic-linker -march=rv64gc)
set(CRUNNER_UTILS_SRC ${WORKLOAD_LIB_DIR}/bbsw/CRunnerUtils/CRunnerUtils.cpp)
set(BBHW_MEM_C ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)
set(BBSW_BAREMETAL_TEST_SRC ${WORKLOAD_LIB_DIR}/bbsw/baremetal/test.c)
set(BBSW_LINUX_TEST_SRC ${WORKLOAD_LIB_DIR}/bbsw/linux/test.c)
set(LLVM_MLIR_EXECUTION_ENGINE_DIR
  ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)
if(NOT EXISTS "${BUCKYBALL_ISA_DIR}/toy/params.h")
  message(FATAL_ERROR "missing ${BUCKYBALL_ISA_DIR}/toy/params.h")
endif()
set(TOY_ISA_DIR ${BUCKYBALL_ISA_DIR}/toy)

function(add_linalg_conv2d_test name main_src)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${main_src})
  set(LLVM_MLIR ${CMAKE_CURRENT_BINARY_DIR}/${name}_llvm.mlir)
  set(LLVM_IR ${CMAKE_CURRENT_BINARY_DIR}/${name}.ll)
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before mlir ctests")
  endif()
  set(BAREMETAL_BIN ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_linalg_${name}-baremetal)
  set(LINUX_BIN ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_linalg_${name}-linux)

  add_custom_command(
    OUTPUT ${OBJ} ${LLVM_MLIR} ${LLVM_IR}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -convert-linalg-to-tile
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
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
      -reconcile-unrealized-casts
      > ${LLVM_MLIR}
    COMMAND ${BUDDY_TRANSLATE} --buddy-to-llvmir ${LLVM_MLIR}
      > ${LLVM_IR}
    COMMAND ${BUDDY_LLC} ${LLVM_IR}
      -filetype=obj -mtriple=riscv64 -O2 -code-model=medium
      -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o ${OBJ}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT} ${BUDDY_TRANSLATE} ${BUDDY_LLC}
    COMMENT "Building ${name}.o from Linalg Conv2D MLIR"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${BAREMETAL_BIN}
    COMMAND ${ELF_CC} ${C_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -I${WORKLOAD_LIB_DIR}
      -I${TOY_ISA_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${BBSW_BAREMETAL_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${BBSW_BAREMETAL_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
    COMMENT "Linking ${BAREMETAL_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${LINUX_BIN}
    COMMAND ${LINUX_CXX} ${LINUX_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -I${WORKLOAD_LIB_DIR}
      -I${TOY_ISA_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN}
      ${BBSW_LINUX_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${BBSW_LINUX_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
    COMMENT "Linking ${LINUX_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  add_custom_target(mlirtest_linalg_${name}_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})

  add_custom_target(mlirtest_linalg_${name}_linux
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})

  add_custom_target(mlirtest_linalg_${name}
    DEPENDS mlirtest_linalg_${name}_linux
      mlirtest_linalg_${name}_baremetal
    COMMENT "Building mlirtest_linalg_${name} for all platforms"
  )

  add_toy_mlir_rushb(
    NAME ${name}
    TARGET_STEM ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_linalg_${name}
    TARGET toy
    CXX
    FRONT_PASSES
      -convert-linalg-to-tile
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
    SOURCES
      ${BBSW_LINUX_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
    INCLUDE_DIRS
      ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      ${WORKLOAD_LIB_DIR}
      ${TOY_ISA_DIR}
    DEPENDS
      ${MAIN_SRC}
      ${BBSW_LINUX_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C})
endfunction()

add_linalg_conv2d_test(conv2d_nhwc_hwcf conv2d_nhwc_main.cpp)
add_linalg_conv2d_test(conv2d_nhwc_fhwc conv2d_nhwc_main.cpp)
add_linalg_conv2d_test(conv2d_nchw_fchw conv2d_nchw_main.cpp)

add_custom_target(linalg-mlir-tests-build ALL DEPENDS
  mlirtest_linalg_conv2d_nhwc_hwcf
  mlirtest_linalg_conv2d_nhwc_fhwc
  mlirtest_linalg_conv2d_nchw_fchw
  COMMENT "Building all Linalg MLIR tests"
  VERBATIM)
