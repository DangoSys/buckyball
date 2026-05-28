set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -DBAREMETAL -Wl,-T,${BBSIM_LD})
set(LINUX_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
set(LINUX_FLAGS -static -Wl,--no-dynamic-linker -march=rv64gc)
set(CRUNNER_UTILS_SRC ${WORKLOAD_LIB_DIR}/bbsw/CRunnerUtils/CRunnerUtils.cpp)
set(LLVM_MLIR_EXECUTION_ENGINE_DIR
  ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)
set(FILECHECK ${LLVM_BUILD_DIR}/bin/FileCheck)

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
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_1_buckyball.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_2_bank_ssa.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_3_physical_banks.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
      -lower-bank-ssa-to-intrinsics
      -lower-buckyball
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_4_buckyball_intrinsics.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
      -lower-bank-ssa-to-intrinsics
      -convert-linalg-to-loops
      -expand-strided-metadata
      -convert-scf-to-cf
      -lower-buckyball
      -convert-arith-to-llvm
      -convert-math-to-llvm
      -finalize-memref-to-llvm
      -convert-func-to-llvm
      -reconcile-unrealized-casts
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_5_llvm.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
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
    ${BUDDY_TRANSLATE} --buddy-to-llvmir
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_6.ll
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
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
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
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

function(add_tile_matmul_contract_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${name}_1_buckyball.mlir)

  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      "-convert-tile-to-buckyball=bank_depth=128"
      > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Tile contract IR for ${name}"
    VERBATIM
  )

  add_custom_target(optest_tile_${name}_contract
    COMMAND ${FILECHECK} ${MLIR_SRC} --input-file=${OUT}
    DEPENDS ${OUT}
    COMMENT "Checking Tile matmul contract for ${name}"
    VERBATIM
  )
endfunction()

function(add_tile_contract_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${name}_bank_ssa.mlir)

  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Tile contract IR for ${name}"
    VERBATIM
  )

  add_custom_target(optest_tile_${name}_contract
    COMMAND ${FILECHECK} ${MLIR_SRC} --input-file=${OUT}
    DEPENDS ${OUT}
    COMMENT "Checking Tile contract for ${name}"
    VERBATIM
  )
endfunction()

add_tile_matmul_test(matmul_127x17_17x127)
add_tile_matmul_contract_test(matmul_16x2048_2048x16_contract)
add_tile_contract_test(conv2d_nhwc_hwcf_contract)

add_custom_target(tile-OpTest-build ALL DEPENDS
  optest_tile_matmul_127x17_17x127
  COMMENT "Building all Tile OpTest workloads"
  VERBATIM)
