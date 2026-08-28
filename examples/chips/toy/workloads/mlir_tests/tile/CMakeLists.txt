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

function(add_tile_matmul_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp)
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before mlir ctests")
  endif()
  set(BAREMETAL_BIN ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_tile_${name}-baremetal)
  set(LINUX_BIN ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_tile_${name}-linux)

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -convert-tile-to-buckyball
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_1_buckyball.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_2_bank_ssa.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      -assign-physical-banks
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_3_physical_banks.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      -assign-physical-banks
      ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS}
      ${BUCKYBALL_LOWER_BUCKYBALL}
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_4_buckyball_intrinsics.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
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
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_5_llvm.mlir
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
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
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir
      > ${CMAKE_CURRENT_BINARY_DIR}/${name}_6.ll
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
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
      -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir |
    ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=${BUCKYBALL_RISCV_MATTR}
      -float-abi=hard -o ${OBJ}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building ${name}.o from Tile MLIR"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${BAREMETAL_BIN}
    COMMAND ${ELF_CC} ${C_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -I${WORKLOAD_LIB_DIR}
      -I${WORKLOAD_LIB_DIR}/bbhw/mem
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
      -I${WORKLOAD_LIB_DIR}/bbhw/mem
      -o ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${CRUNNER_UTILS_SRC}
      ${BBHW_MEM_C}
    COMMENT "Linking ${LINUX_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  add_custom_target(mlirtest_tile_${name}_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})

  add_custom_target(mlirtest_tile_${name}_linux
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})

  add_custom_target(mlirtest_tile_${name}
    DEPENDS mlirtest_tile_${name}_linux
      mlirtest_tile_${name}_baremetal
    COMMENT "Building mlirtest_tile_${name} for all platforms"
  )

  add_toy_mlir_rushb(
    NAME ${name}
    TARGET_STEM ${BUCKYBALL_WORKLOAD_CHIP}_mlirtest_tile_${name}
    TARGET toy
    CXX
    FRONT_PASSES -convert-tile-to-buckyball -lower-buckyball-to-bank-ssa
    SOURCES ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C} ${MAIN_SRC}
    INCLUDE_DIRS
      ${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      ${WORKLOAD_LIB_DIR}
      ${WORKLOAD_LIB_DIR}/bbhw/mem
    DEPENDS ${MAIN_SRC} ${CRUNNER_UTILS_SRC} ${BBHW_MEM_C})
endfunction()

function(add_tile_matmul_contract_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${name}_1_buckyball.mlir)

  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      "-convert-tile-to-buckyball=bank_depth=128"
      > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Tile contract IR for ${name}"
    VERBATIM
  )

  add_custom_target(mlirtest_tile_${name}_contract
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
    COMMAND ${BUDDY_OPT} ${MLIR_SRC} --target=toy
      -convert-tile-to-buckyball
      -lower-buckyball-to-bank-ssa
      > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Tile contract IR for ${name}"
    VERBATIM
  )

  add_custom_target(mlirtest_tile_${name}_contract
    COMMAND ${FILECHECK} ${MLIR_SRC} --input-file=${OUT}
    DEPENDS ${OUT}
    COMMENT "Checking Tile contract for ${name}"
    VERBATIM
  )
endfunction()

add_tile_contract_test(conv2d_nhwc_hwcf_contract)
add_tile_contract_test(conv2d_nhwc_hwcf_pad_contract)

add_custom_target(tile-mlir-tests-build ALL
  COMMENT "Building all Tile MLIR tests"
  VERBATIM)
