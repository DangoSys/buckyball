set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -DBAREMETAL -Wl,-T,${BBSIM_LD})
set(LINUX_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
set(LINUX_FLAGS -static -Wl,--no-dynamic-linker -march=rv64gc)
set(CRUNNER_UTILS_SRC ${WORKLOAD_LIB_DIR}/bbsw/CRunnerUtils/CRunnerUtils.cpp)
set(LLVM_MLIR_EXECUTION_ENGINE_DIR
  ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)

function(add_tile_matmul_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.c)
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  set(BAREMETAL_BIN optest_tile_${name}_singlecore-baremetal)
  set(LINUX_BIN optest_tile_${name}-linux)

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=32"
      -lower-bank-ssa-to-intrinsics
      -convert-linalg-to-loops
      -expand-strided-metadata
      -convert-scf-to-cf
      -lower-buckyball
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir |
    ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D
      -float-abi=hard -o ${OBJ}
    DEPENDS ${MLIR_SRC}
    COMMENT "Building ${name}.o from Tile MLIR"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${BAREMETAL_BIN}
    COMMAND ${ELF_CC} ${C_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${CRUNNER_UTILS_SRC}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${CRUNNER_UTILS_SRC}
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
      -o ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN}
      ${CRUNNER_UTILS_SRC}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${CRUNNER_UTILS_SRC}
    COMMENT "Linking ${LINUX_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  add_custom_target(optest_tile_${name}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})

  add_custom_target(optest_tile_${name}_linux
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})

  add_custom_target(optest_tile_${name}
    DEPENDS optest_tile_${name}_linux
      optest_tile_${name}_singlecore_baremetal
    COMMENT "Building optest_tile_${name} for all platforms"
  )
endfunction()

add_tile_matmul_test(matmul_127x17_17x127)
