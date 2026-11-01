set(CORE_PLUGIN "${DESIGN_CORE_BUILD_goban}/libcore-compiler.so")
set(CORE_OPT ${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=${CORE_PLUGIN} --load-pass-plugin=${CORE_PLUGIN})
set(ACC_OPT ${CORE_OPT} --buckyball-design-pb=${DESIGN_PB} --buckyball-target=matmul)
set(HART ${DESIGN_CORE_HART_goban})
set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_int8_tiles")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/int8_tiles")
add_custom_command(
  OUTPUT ${OUT}/int8_tiles.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/int8_tiles.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB}" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${ACC_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=goban" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/int8_tiles.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/int8_tiles.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/int8_tiles.o
  DEPENDS int8_tiles.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/int8_tiles.c ${OUT}/int8_tiles.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS int8_tiles.c ${OUT}/int8_tiles.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_bias_tail")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/bias_tail")
add_custom_command(
  OUTPUT ${OUT}/bias_tail.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/bias_tail.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB}" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${ACC_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=goban" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/bias_tail.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/bias_tail.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/bias_tail.o
  DEPENDS bias_tail.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/bias_tail.c ${OUT}/bias_tail.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS bias_tail.c ${OUT}/bias_tail.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-core${HART}-mlirtest-compiler_batched_int8")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/batched_int8")
add_custom_command(
  OUTPUT ${OUT}/batched_int8.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CORE_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/batched_int8.mlir
    "-convert-linalg-to-core=design=${DESIGN_PB}" -o ${OUT}/core.mlir
  COMMAND ${ACC_OPT} ${OUT}/core.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${ACC_OPT} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${ACC_OPT} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${ACC_OPT} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${ACC_OPT} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${CORE_OPT} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${CORE_OPT} ${OUT}/addresses.mlir
    "-lower-core=design=${DESIGN_PB} core=goban" -o ${OUT}/engine.mlir
  COMMAND ${CORE_OPT} ${OUT}/engine.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${CORE_OPT} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/batched_int8.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/batched_int8.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/batched_int8.o
  DEPENDS batched_int8.mlir ${CORE_PLUGIN}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/batched_int8.c ${OUT}/batched_int8.o
    ${DESIGN_THREAD_BUILD_matmul}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS batched_int8.c ${OUT}/batched_int8.o ${DESIGN_THREAD_BUILD_matmul}/loader.o
    runtime-baremetal engine
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)
