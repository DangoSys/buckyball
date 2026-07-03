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

function(add_toy_mlir_matmul_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.c)
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  set(BAREMETAL_BIN optest_buckyball_${name}_singlecore-baremetal)
  set(BAREMETAL_TARGET optest_buckyball_${name}_singlecore_baremetal)
  set(LINUX_BIN optest_buckyball_${name}-linux)
  set(LINUX_TARGET optest_buckyball_${name}_linux)

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -lower-buckyball-to-bank-ssa
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_1_bank_ssa.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_2_physical_banks.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
      ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
      ${BUCKYBALL_LOWER_BUCKYBALL}
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_3_buckyball_intrinsics.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
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
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_4_llvm.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
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
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_5.ll
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -lower-buckyball-to-bank-ssa
      "--assign-physical-banks=bank_num=16"
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
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir |
    ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -O2 -code-model=medium
      -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o ${OBJ}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building ${name}.o from Toy Buckyball MLIR"
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

  add_custom_target(${BAREMETAL_TARGET}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})
  add_custom_target(${LINUX_TARGET}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})
  add_custom_target(optest_buckyball_${name}
    DEPENDS ${BAREMETAL_TARGET} ${LINUX_TARGET})
endfunction()

function(add_toy_mlir_matmul_contract_test name)
  add_toy_mlir_matmul_test(${name})
  add_custom_target(optest_buckyball_${name}_contract
    COMMAND ${FILECHECK}
      ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir
      --input-file=${CMAKE_CURRENT_BINARY_DIR}/${name}_1_bank_ssa.mlir
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.o
    COMMENT "Checking Toy Buckyball matmul ISA contract for ${name}"
    VERBATIM)
endfunction()

add_toy_mlir_matmul_contract_test(matmul_16x64_64x16)
add_toy_mlir_matmul_test(matmul_64x16_16x64)
add_toy_mlir_matmul_test(matmul_1024x16_16x1024)
add_toy_mlir_matmul_test(matmul_1x128_128x96)

add_dependencies(toy-mlir-tests-build
  optest_buckyball_matmul_16x64_64x16
  optest_buckyball_matmul_64x16_16x64
  optest_buckyball_matmul_1024x16_16x1024
  optest_buckyball_matmul_1x128_128x96)
