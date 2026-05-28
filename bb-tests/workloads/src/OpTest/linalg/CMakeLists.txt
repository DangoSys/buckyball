set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -DBAREMETAL -Wl,-T,${BBSIM_LD})
set(LINUX_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
set(LINUX_FLAGS -static -Wl,--no-dynamic-linker -march=rv64gc)
set(CRUNNER_UTILS_SRC ${WORKLOAD_LIB_DIR}/bbsw/CRunnerUtils/CRunnerUtils.cpp)
set(BBSW_BAREMETAL_TEST_SRC ${WORKLOAD_LIB_DIR}/bbsw/baremetal/test.c)
set(BBSW_LINUX_TEST_SRC ${WORKLOAD_LIB_DIR}/bbsw/linux/test.c)
set(LLVM_MLIR_EXECUTION_ENGINE_DIR
  ${BUDDY_MLIR_DIR}/llvm/mlir/include/mlir/ExecutionEngine)
set(FILECHECK ${LLVM_BUILD_DIR}/bin/FileCheck)

function(add_linalg_contract_test name)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${name}_tile.mlir)

  add_custom_command(
    OUTPUT ${OUT}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-linalg-to-tile
      > ${OUT}
    DEPENDS ${MLIR_SRC} ${BUDDY_OPT}
    COMMENT "Building Linalg contract IR for ${name}"
    VERBATIM
  )

  add_custom_target(optest_linalg_${name}_contract
    COMMAND ${FILECHECK} ${MLIR_SRC} --input-file=${OUT}
    DEPENDS ${OUT}
    COMMENT "Checking Linalg contract for ${name}"
    VERBATIM
  )
endfunction()

function(add_linalg_conv2d_test name main_src)
  set(MLIR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir)
  set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${main_src})
  set(OBJ ${CMAKE_CURRENT_BINARY_DIR}/${name}.o)
  set(BAREMETAL_BIN optest_linalg_${name}_singlecore-baremetal)
  set(LINUX_BIN optest_linalg_${name}-linux)

  add_custom_command(
    OUTPUT ${OBJ}
    COMMAND ${BUDDY_OPT} ${MLIR_SRC}
      -convert-linalg-to-tile
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
    COMMENT "Building ${name}.o from Linalg Conv2D MLIR"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${BAREMETAL_BIN}
    COMMAND ${ELF_CC} ${C_FLAGS}
      -I${LLVM_MLIR_EXECUTION_ENGINE_DIR}
      -I${WORKLOAD_LIB_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${BBSW_BAREMETAL_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${BBSW_BAREMETAL_TEST_SRC}
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
      -I${WORKLOAD_LIB_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN}
      ${BBSW_LINUX_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
      ${MAIN_SRC}
      ${OBJ}
    DEPENDS ${OBJ}
      ${MAIN_SRC}
      ${BBSW_LINUX_TEST_SRC}
      ${CRUNNER_UTILS_SRC}
    COMMENT "Linking ${LINUX_BIN}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  add_custom_target(optest_linalg_${name}_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BAREMETAL_BIN})

  add_custom_target(optest_linalg_${name}_linux
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${LINUX_BIN})

  add_custom_target(optest_linalg_${name}
    DEPENDS optest_linalg_${name}_linux
      optest_linalg_${name}_singlecore_baremetal
    COMMENT "Building optest_linalg_${name} for all platforms"
  )
endfunction()

# Compile matmul.mlir -> matmul.o
add_custom_command(
    OUTPUT matmul.o
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -convert-linalg-to-tile
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
    ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
        -o ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
    COMMENT "Building matmul.o from Linalg MLIR"
    VERBATIM
)

# Link baremetal binary
add_custom_command(
    OUTPUT optest_linalg_matmul_singlecore-baremetal
    COMMAND ${ELF_CC} ${C_FLAGS}
        -o ${CMAKE_CURRENT_BINARY_DIR}/optest_linalg_matmul_singlecore-baremetal
        ${BBSW_BAREMETAL_DIR}/crt0.S
        ${BBSW_BAREMETAL_DIR}/syscalls.c
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_main.c
        ${BBSW_BAREMETAL_DIR}/crt0.S
        ${BBSW_BAREMETAL_DIR}/syscalls.c
    COMMENT "Linking optest_linalg_matmul_singlecore-baremetal"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

add_custom_target(optest_linalg_matmul_singlecore_baremetal ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optest_linalg_matmul_singlecore-baremetal
)

add_linalg_contract_test(conv2d_nhwc_hwcf_contract)
add_linalg_contract_test(conv2d_nhwc_fhwc_contract)
add_linalg_contract_test(conv2d_nchw_fchw_contract)

add_linalg_conv2d_test(conv2d_nhwc_hwcf conv2d_nhwc_main.c)
add_linalg_conv2d_test(conv2d_nhwc_fhwc conv2d_nhwc_main.c)
add_linalg_conv2d_test(conv2d_nchw_fchw conv2d_nchw_main.c)

add_custom_target(linalg-OpTest-build ALL DEPENDS
  optest_linalg_matmul_singlecore_baremetal
  optest_linalg_conv2d_nhwc_hwcf
  optest_linalg_conv2d_nhwc_fhwc
  optest_linalg_conv2d_nchw_fchw
  COMMENT "Building all Linalg OpTest workloads"
  VERBATIM)
