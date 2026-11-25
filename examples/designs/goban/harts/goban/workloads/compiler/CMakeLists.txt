# The operator image of the GP domain: the operators and their entry `operator`
# that convert-linalg-to-acc leaves in acc.mlir, linked by operators.ld.
set(OPERATOR_LINKER "${BUCKYBALL_REPO_ROOT}/examples/designs/goban/harts/goban/stack/operators.ld")
math(EXPR OPERATOR_TEXT_BYTES "${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
find_program(RVV_LD riscv64-unknown-elf-ld REQUIRED)
find_program(LLVM_OPT opt PATHS "${DESIGN_REPO}/stack/thirdparty/buddy-mlir/llvm/build/bin" NO_DEFAULT_PATH REQUIRED)
set(HART ${DESIGN_HART_HART_goban})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_padded_copy")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/padded_copy")
add_custom_command(
  OUTPUT ${OUT}/padded_copy.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/padded_copy.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/padded_copy.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/padded_copy.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/padded_copy.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS padded_copy.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/padded_copy.cpp ${OUT}/padded_copy.o
    ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS padded_copy.cpp ${OUT}/padded_copy.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_int8_tiles")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/int8_tiles")
add_custom_command(
  OUTPUT ${OUT}/int8_tiles.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/int8_tiles.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/int8_tiles.mlir --check-prefixes=REUSE,ONE
    --input-file=${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,compute_one,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/int8_tiles.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/int8_tiles.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/int8_tiles.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS int8_tiles.mlir ${DESIGN_HART_PLUGIN_goban} ${OPERATOR_LINKER}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/stack/models/lib
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/int8_tiles.cpp ${OUT}/int8_tiles.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS int8_tiles.cpp ${OUT}/int8_tiles.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(MULTIPLY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rvv_multiply")
find_program(MULTIPLY_CXX clang++ PATHS "${DESIGN_REPO}/result/bin" NO_DEFAULT_PATH REQUIRED)
set(MULTIPLY_LINKER "${DESIGN_REPO}/bb-tests/workloads/lib/bbsw/thread/rvv/operator.ld")
set(MULTIPLY_DEFS ${DESIGN_TARGET_DEFS_matmul})
list(TRANSFORM MULTIPLY_DEFS PREPEND "-D")
add_custom_command(
  OUTPUT ${MULTIPLY_DIR}/rvv_multiply.image
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MULTIPLY_DIR}
  COMMAND ${MULTIPLY_CXX} --target=riscv64 -march=rv64imf_zve64f_zvl128b
    -mabi=lp64f -mcmodel=medany -ffreestanding -O2
    -ffunction-sections -fdata-sections -c ${CMAKE_CURRENT_SOURCE_DIR}/rvv_multiply_operator.cpp
    -o ${MULTIPLY_DIR}/rvv_multiply.o
  COMMAND ${RVV_LD} -m elf64lriscv --gc-sections -e multiply --defsym=operator_entry=multiply
    -T ${MULTIPLY_LINKER} ${MULTIPLY_DIR}/rvv_multiply.o -o ${MULTIPLY_DIR}/rvv_multiply.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${MULTIPLY_DIR}/rvv_multiply.elf ${MULTIPLY_DIR}/rvv_multiply.image
  DEPENDS rvv_multiply_operator.cpp ${MULTIPLY_LINKER}
  VERBATIM)
set(TEST "${DESIGN_NAME}-hart${HART}-ctest-rvv_multiply")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -DBUCKYBALL_STREAM ${MULTIPLY_DEFS}
    -I${DESIGN_REPO}/bb-tests/workloads/lib
    -I${DESIGN_REPO}/bb-tests/workloads/lib/bbsw/thread
    -I${DESIGN_REPO}/bb-tests/workloads/lib/bbhw/isa -I${DESIGN_REPO}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    --embed-dir=${MULTIPLY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_multiply.cpp
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_multiply.cpp ${MULTIPLY_DIR}/rvv_multiply.image
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_bias_tail")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/bias_tail")
add_custom_command(
  OUTPUT ${OUT}/bias_tail.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/bias_tail.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/bias_tail.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/bias_tail.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/bias_tail.o
  DEPENDS bias_tail.mlir ${DESIGN_HART_PLUGIN_goban}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/bias_tail.c ${OUT}/bias_tail.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS bias_tail.c ${OUT}/bias_tail.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_batched_int8")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/batched_int8")
add_custom_command(
  OUTPUT ${OUT}/batched_int8.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/batched_int8.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/batched_int8.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/batched_int8.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/batched_int8.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS batched_int8.mlir ${DESIGN_HART_PLUGIN_goban} ${OPERATOR_LINKER}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/batched_int8.cpp ${OUT}/batched_int8.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS batched_int8.cpp ${OUT}/batched_int8.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_rows")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_rows")
add_custom_command(
  OUTPUT ${OUT}/rvv_rows.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_rows.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_rows.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_rows.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_rows.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_rows.cpp ${OUT}/rvv_rows.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_rows.cpp ${OUT}/rvv_rows.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_norm")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_norm")
add_custom_command(
  OUTPUT ${OUT}/rvv_norm.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_norm.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_norm.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_norm.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_norm.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_norm.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_norm.c ${OUT}/rvv_norm.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_norm.c ${OUT}/rvv_norm.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_erf")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_erf")
add_custom_command(
  OUTPUT ${OUT}/rvv_erf.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_erf.mlir
    -convert-elementwise-to-linalg -o ${OUT}/linalg.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/linalg.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_erf.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_erf.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_erf.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_erf.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_erf.c ${OUT}/rvv_erf.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_erf.c ${OUT}/rvv_erf.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_transpose_f32")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/transpose_f32")
add_custom_command(
  OUTPUT ${OUT}/transpose_f32.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/transpose_f32.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/transpose_f32.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/transpose_f32.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/transpose_f32.o
  DEPENDS transpose_f32.mlir ${DESIGN_HART_PLUGIN_goban}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/transpose_f32.c ${OUT}/transpose_f32.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS transpose_f32.c ${OUT}/transpose_f32.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_reduce")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_reduce")
add_custom_command(
  OUTPUT ${OUT}/rvv_reduce.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_reduce.mlir
    -convert-elementwise-to-linalg -o ${OUT}/linalg.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/linalg.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_reduce.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_reduce.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_reduce.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_reduce.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_reduce.c ${OUT}/rvv_reduce.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_reduce.c ${OUT}/rvv_reduce.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_max")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_max")
add_custom_command(
  OUTPUT ${OUT}/rvv_max.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_max.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -lower-acc -expand-strided-metadata -lower-affine
    -canonicalize "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_max.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_max.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_max.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_max.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_max.cpp ${OUT}/rvv_max.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_max.cpp ${OUT}/rvv_max.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_matmul_rows")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/matmul_rows")
add_custom_command(
  OUTPUT ${OUT}/matmul_rows.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/matmul_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/matmul_rows.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/matmul_rows.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/matmul_rows.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS matmul_rows.mlir ${DESIGN_HART_PLUGIN_goban} ${OPERATOR_LINKER}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_rows.cpp ${OUT}/matmul_rows.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS matmul_rows.cpp ${OUT}/matmul_rows.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quantized_rows")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/quantized_rows")
add_custom_command(
  OUTPUT ${OUT}/quantized_rows.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/quantized_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/quantized_rows.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/quantized_rows.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/quantized_rows.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS quantized_rows.mlir ${DESIGN_HART_PLUGIN_goban} ${OPERATOR_LINKER} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized_rows.cpp ${OUT}/quantized_rows.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS quantized_rows.cpp ${OUT}/quantized_rows.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

find_program(GATHER_RVV_CXX clang++ PATHS "${DESIGN_REPO}/result/bin" NO_DEFAULT_PATH REQUIRED)
set(GATHER_DIR "${CMAKE_CURRENT_BINARY_DIR}/rvv_gather")
set(GATHER_DEFS ${DESIGN_TARGET_DEFS_matmul})
list(TRANSFORM GATHER_DEFS PREPEND "-D")
set(GATHER_LINKER "${DESIGN_REPO}/bb-tests/workloads/lib/bbsw/thread/rvv/operator.ld")
add_custom_command(
  OUTPUT ${GATHER_DIR}/rvv_gather.image
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GATHER_DIR}
  COMMAND ${GATHER_RVV_CXX} --target=riscv64 -march=rv64imf_zve32f_zvl128b
    -mabi=lp64f -mcmodel=medany -ffreestanding -O2 -ffp-contract=off
    -ffunction-sections -fdata-sections -c ${CMAKE_CURRENT_SOURCE_DIR}/rvv_gather_operator.cpp
    -o ${GATHER_DIR}/rvv_gather.o
  COMMAND ${RVV_LD} -m elf64lriscv --gc-sections -e gather --defsym=operator_entry=gather
    -T ${GATHER_LINKER} ${GATHER_DIR}/rvv_gather.o -o ${GATHER_DIR}/rvv_gather.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${GATHER_DIR}/rvv_gather.elf ${GATHER_DIR}/rvv_gather.image
  DEPENDS rvv_gather_operator.cpp ${GATHER_LINKER}
  VERBATIM)
set(TEST "${DESIGN_NAME}-hart${HART}-ctest-rvv_gather")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -DBUCKYBALL_STREAM ${GATHER_DEFS}
    -I${DESIGN_REPO}/bb-tests/workloads/lib
    -I${DESIGN_REPO}/bb-tests/workloads/lib/bbsw/thread
    -I${DESIGN_REPO}/bb-tests/workloads/lib/bbhw/isa -I${DESIGN_REPO}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    --embed-dir=${GATHER_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_gather.cpp
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_gather.cpp ${GATHER_DIR}/rvv_gather.image
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

find_program(STRIDE_RVV_CXX clang++ PATHS "${DESIGN_REPO}/result/bin" NO_DEFAULT_PATH REQUIRED)
set(STRIDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/rvv_stride")
set(STRIDE_DEFS ${DESIGN_TARGET_DEFS_matmul})
list(TRANSFORM STRIDE_DEFS PREPEND "-D")
set(STRIDE_LINKER "${DESIGN_REPO}/bb-tests/workloads/lib/bbsw/thread/rvv/operator.ld")
add_custom_command(
  OUTPUT ${STRIDE_DIR}/rvv_stride.image
  COMMAND ${CMAKE_COMMAND} -E make_directory ${STRIDE_DIR}
  COMMAND ${STRIDE_RVV_CXX} --target=riscv64 -march=rv64imf_zve64f_zvl128b
    -mabi=lp64f -mcmodel=medany -ffreestanding -O2 -ffp-contract=off
    -ffunction-sections -fdata-sections -c ${CMAKE_CURRENT_SOURCE_DIR}/rvv_stride_operator.cpp
    -o ${STRIDE_DIR}/rvv_stride.o
  COMMAND ${RVV_LD} -m elf64lriscv --gc-sections -e strided --defsym=operator_entry=strided
    -T ${STRIDE_LINKER} ${STRIDE_DIR}/rvv_stride.o -o ${STRIDE_DIR}/rvv_stride.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${STRIDE_DIR}/rvv_stride.elf ${STRIDE_DIR}/rvv_stride.image
  DEPENDS rvv_stride_operator.cpp ${STRIDE_LINKER}
  VERBATIM)
set(TEST "${DESIGN_NAME}-hart${HART}-ctest-rvv_stride")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    -DBUCKYBALL_STREAM ${STRIDE_DEFS}
    -I${DESIGN_REPO}/bb-tests/workloads/lib
    -I${DESIGN_REPO}/bb-tests/workloads/lib/bbsw/thread
    -I${DESIGN_REPO}/bb-tests/workloads/lib/bbhw/isa -I${DESIGN_REPO}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    --embed-dir=${STRIDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_stride.cpp
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_stride.cpp ${STRIDE_DIR}/rvv_stride.image
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_exp_underflow")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/exp_underflow")
add_custom_command(
  OUTPUT ${OUT}/exp_underflow.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/exp_underflow.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}"
    -convert-math-to-libm -convert-arith-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/exp_underflow.ll
  COMMAND ${LLVM_OPT} ${OUT}/exp_underflow.ll -passes=default<O2> -S
    -o ${OUT}/optimized.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/optimized.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/exp_underflow.o
  DEPENDS exp_underflow.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CXX} ${CTEST_C_FLAGS} -std=c++20 -fno-exceptions -fno-rtti
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/exp_underflow.cpp ${OUT}/exp_underflow.o
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS exp_underflow.cpp ${OUT}/exp_underflow.o
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_strided")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_strided")
add_custom_command(
  OUTPUT ${OUT}/rvv_strided.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_strided.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    "-pass-pipeline=builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_strided.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_strided.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_strided.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_strided.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_strided.c ${OUT}/rvv_strided.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_strided.c ${OUT}/rvv_strided.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_exp")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_exp")
add_custom_command(
  OUTPUT ${OUT}/rvv_exp.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_exp.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_exp.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_exp.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_exp.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_exp.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_exp.c ${OUT}/rvv_exp.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_exp.c ${OUT}/rvv_exp.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_transpose_blocks")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/transpose_blocks")
add_custom_command(
  OUTPUT ${OUT}/transpose_blocks.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/transpose_blocks.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/transpose_blocks.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/transpose_blocks.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/transpose_blocks.o
  DEPENDS transpose_blocks.mlir ${DESIGN_HART_PLUGIN_goban}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/transpose_blocks.c ${OUT}/transpose_blocks.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS transpose_blocks.c ${OUT}/transpose_blocks.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rvv_slice_pair")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rvv_slice_pair")
add_custom_command(
  OUTPUT ${OUT}/rvv_slice_pair.o ${OUT}/operators.embed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_goban} ${CMAKE_CURRENT_SOURCE_DIR}/rvv_slice_pair.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/host.mlir
    -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_matmul} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=goban" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rvv_slice_pair.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rvv_slice_pair.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rvv_slice_pair.o
  COMMAND ${DESIGN_HART_OPT_goban} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/operators-llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators-llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${LLVM_OPT} ${OUT}/operators.ll -mtriple=riscv64 -mattr=+m,+f,+d,+zve64f,+zvl128b
    "-passes=default<O2>" -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=+m,+f,+d,+zve64f,+zvl128b
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_matmul_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${RVV_LD} -m elf64lriscv --defsym=operator_text_bytes=${OPERATOR_TEXT_BYTES}
    -T ${OPERATOR_LINKER} ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT}
    ${ELF_OBJCOPY} -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS rvv_slice_pair.mlir ${DESIGN_HART_PLUGIN_goban} ${DESIGN_PB} ${OPERATOR_LINKER}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rvv_slice_pair.c ${OUT}/rvv_slice_pair.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rvv_slice_pair.c ${OUT}/rvv_slice_pair.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)
