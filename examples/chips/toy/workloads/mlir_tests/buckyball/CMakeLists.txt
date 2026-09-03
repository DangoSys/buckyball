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
set(FILECHECK ${LLVM_BUILD_DIR}/bin/FileCheck)
if(NOT EXISTS "${BUCKYBALL_ISA_DIR}/toy/params.h")
  message(FATAL_ERROR "missing ${BUCKYBALL_ISA_DIR}/toy/params.h")
endif()
set(TOY_ISA_DIR ${BUCKYBALL_ISA_DIR}/toy)

function(add_toy_mlir_matmul_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp)
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before mlir ctests")
  endif()
  set(BAREMETAL_BIN ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_buckyball_${name}-baremetal)
  set(BAREMETAL_TARGET mlirtest_buckyball_${name}_baremetal)
  set(LINUX_BIN ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_buckyball_${name}-linux)
  set(LINUX_TARGET mlirtest_buckyball_${name}_linux)

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -lower-buckyball-to-bank-ssa
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_1_bank_ssa.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -lower-buckyball-to-bank-ssa
      -assign-physical-banks
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_2_physical_banks.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -lower-buckyball-to-bank-ssa
      -assign-physical-banks
      ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
      ${BUCKYBALL_LOWER_BUCKYBALL}
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_3_buckyball_intrinsics.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
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
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_4_llvm.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
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
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_5.ll
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
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
      -I${WORKLOAD_LIB_DIR}
      -I${TOY_ISA_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${TOY_ISA_DIR}/params.h
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
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${TOY_ISA_DIR}/params.h
    COMMENT "Linking ${LINUX_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  add_custom_target(${BAREMETAL_TARGET}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})
  add_custom_target(${LINUX_TARGET}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})
  add_custom_target(mlirtest_buckyball_${name}
    DEPENDS ${BAREMETAL_TARGET} ${LINUX_TARGET})

  add_toy_mlir_rushb(
    NAME ${name}
    TARGET_STEM ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_buckyball_${name}
    TARGET toy
    CXX
    FRONT_PASSES -lower-buckyball-to-bank-ssa
    SOURCES ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C} ${MAIN_SRC}
    INCLUDE_DIRS
      ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      ${WORKLOAD_LIB_DIR}
      ${TOY_ISA_DIR}
    DEPENDS ${MAIN_SRC} ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C} ${TOY_ISA_DIR}/params.h)
endfunction()

function(add_toy_mlir_matmul_contract_test name)
  add_toy_mlir_matmul_test(${name})
  add_custom_target(mlirtest_buckyball_${name}_contract
    COMMAND ${FILECHECK}
      ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir
      --input-file=${CMAKE_CURRENT_BINARY_DIR}/${name}_1_bank_ssa.mlir
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.o
    COMMENT "Checking Toy Buckyball matmul ISA contract for ${name}"
    VERBATIM)
endfunction()

add_custom_target(buckyball-mlir-tests-build ALL
  COMMENT "Building Toy Buckyball MLIR tests"
  VERBATIM)
