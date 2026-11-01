set(CORE_PLUGIN "${DESIGN_DEVICE_BUILD}/core/compiler/libcore-compiler.so")
set(CORE_OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${CORE_PLUGIN} --load-pass-plugin=${CORE_PLUGIN})
set(ACC_OPT ${CORE_OPT} --buckyball-design-pb=${DESIGN_PB})
set(HART ${DESIGN_CORE_HART_poly})
set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_dynamic_mxint8")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/dynamic_mxint8")
add_custom_command(
  OUTPUT ${OUT}/dynamic_mxint8.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mxint8.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB} core=poly" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir --buckyball-target=attention
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_acc.mlir --buckyball-target=ffn
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${CORE_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir --buckyball-target=attention
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_banks.mlir --buckyball-target=attention
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_instructions.mlir --buckyball-target=ffn
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/ffn_banks.mlir --buckyball-target=ffn
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=poly" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/dynamic_mxint8.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/dynamic_mxint8.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/dynamic_mxint8.o
  DEPENDS dynamic_mxint8.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mxint8.c ${OUT}/dynamic_mxint8.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS dynamic_mxint8.c ${OUT}/dynamic_mxint8.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_mixed_attention")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/mixed_attention")
add_custom_command(
  OUTPUT ${OUT}/mixed_attention.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/mixed_attention.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB} core=poly" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir --buckyball-target=attention
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_acc.mlir --buckyball-target=ffn
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${CORE_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir --buckyball-target=attention
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_banks.mlir --buckyball-target=attention
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_instructions.mlir --buckyball-target=ffn
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/ffn_banks.mlir --buckyball-target=ffn
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=poly" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/mixed_attention.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/mixed_attention.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/mixed_attention.o
  DEPENDS mixed_attention.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/mixed_attention.c ${OUT}/mixed_attention.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS mixed_attention.c ${OUT}/mixed_attention.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_quantized_key")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/quantized_key")
add_custom_command(
  OUTPUT ${OUT}/quantized_key.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/quantized_key.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB} core=poly" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir --buckyball-target=attention
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_acc.mlir --buckyball-target=ffn
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${CORE_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir --buckyball-target=attention
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_banks.mlir --buckyball-target=attention
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_instructions.mlir --buckyball-target=ffn
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/ffn_banks.mlir --buckyball-target=ffn
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=poly" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/quantized_key.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/quantized_key.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/quantized_key.o
  DEPENDS quantized_key.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized_key.c ${OUT}/quantized_key.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS quantized_key.c ${OUT}/quantized_key.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_dynamic_attention")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/dynamic_attention")
add_custom_command(
  OUTPUT ${OUT}/dynamic_attention.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_attention.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB} core=poly" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir --buckyball-target=attention
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_acc.mlir --buckyball-target=ffn
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${CORE_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir --buckyball-target=attention
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_banks.mlir --buckyball-target=attention
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_instructions.mlir --buckyball-target=ffn
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/ffn_banks.mlir --buckyball-target=ffn
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=poly" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/dynamic_attention.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/dynamic_attention.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/dynamic_attention.o
  DEPENDS dynamic_attention.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_attention.c ${OUT}/dynamic_attention.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS dynamic_attention.c ${OUT}/dynamic_attention.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)


set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_packed_weights")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/packed_weights")
add_custom_command(
  OUTPUT ${OUT}/packed_weights.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB} core=poly" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir --buckyball-target=attention
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_acc.mlir --buckyball-target=ffn
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${CORE_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir --buckyball-target=attention
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_banks.mlir --buckyball-target=attention
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${ACC_OPT} ${OUT}/attention_instructions.mlir --buckyball-target=ffn
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/ffn_banks.mlir --buckyball-target=ffn
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=poly" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/packed_weights.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/packed_weights.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/packed_weights.o
  DEPENDS packed_weights.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.c ${OUT}/packed_weights.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS packed_weights.c ${OUT}/packed_weights.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)
