set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_mxint8")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/mxint8")
add_custom_command(
  OUTPUT ${OUT}/mxint8.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/mxint8.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/mxint8.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/mxint8.o
  DEPENDS mxint8.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/mxint8.c ${OUT}/mxint8.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS mxint8.c ${OUT}/mxint8.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_linear_bias")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/linear_bias")
add_custom_command(
  OUTPUT ${OUT}/linear_bias.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/linear_bias.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/linear_bias.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/linear_bias.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/linear_bias.o
  DEPENDS linear_bias.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_bias.c ${OUT}/linear_bias.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS linear_bias.c ${OUT}/linear_bias.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_attention_tail")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/attention_tail")
add_custom_command(
  OUTPUT ${OUT}/attention_tail.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/attention_tail.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/attention_tail.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/attention_tail.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/attention_tail.o
  DEPENDS attention_tail.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/attention_tail.c ${OUT}/attention_tail.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS attention_tail.c ${OUT}/attention_tail.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_packed_weights")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/packed_weights")
add_custom_command(
  OUTPUT ${OUT}/packed_weights.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/packed_weights.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/packed_weights.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/packed_weights.o
  DEPENDS packed_weights.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.c ${OUT}/packed_weights.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS packed_weights.c ${OUT}/packed_weights.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_softmax")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/softmax")
add_custom_command(
  OUTPUT ${OUT}/softmax.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/softmax.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/softmax.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/softmax.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/softmax.o
  DEPENDS softmax.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.c ${OUT}/softmax.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS softmax.c ${OUT}/softmax.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_softmax_tail")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/softmax_tail")
add_custom_command(
  OUTPUT ${OUT}/softmax_tail.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/softmax_tail.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/softmax_tail.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/softmax_tail.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/softmax_tail.o
  DEPENDS softmax_tail.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax_tail.c ${OUT}/softmax_tail.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS softmax_tail.c ${OUT}/softmax_tail.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_softmax_scale")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/softmax_scale")
add_custom_command(
  OUTPUT ${OUT}/softmax_scale.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/softmax_scale.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/softmax_scale.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/softmax_scale.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/softmax_scale.o
  DEPENDS softmax_scale.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax_scale.c ${OUT}/softmax_scale.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS softmax_scale.c ${OUT}/softmax_scale.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_layernorm")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/layernorm")
add_custom_command(
  OUTPUT ${OUT}/layernorm.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/layernorm.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/layernorm.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/layernorm.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/layernorm.o
  DEPENDS layernorm.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm.c ${OUT}/layernorm.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS layernorm.c ${OUT}/layernorm.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_doubled_weights")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/doubled_weights")
add_custom_command(
  OUTPUT ${OUT}/doubled_weights.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/doubled_weights.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/doubled_weights.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/doubled_weights.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/doubled_weights.o
  DEPENDS doubled_weights.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/doubled_weights.c ${OUT}/doubled_weights.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS doubled_weights.c ${OUT}/doubled_weights.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_doubled_blocks")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/doubled_blocks")
add_custom_command(
  OUTPUT ${OUT}/doubled_blocks.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/doubled_blocks.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/doubled_blocks.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/doubled_blocks.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/doubled_blocks.o
  DEPENDS doubled_blocks.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/doubled_blocks.c ${OUT}/doubled_blocks.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS doubled_blocks.c ${OUT}/doubled_blocks.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_residual_weights")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/residual_weights")
add_custom_command(
  OUTPUT ${OUT}/residual_weights.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/residual_weights.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/residual_weights.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/residual_weights.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/residual_weights.o
  DEPENDS residual_weights.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/residual_weights.c ${OUT}/residual_weights.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS residual_weights.c ${OUT}/residual_weights.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_residual_blocks")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/residual_blocks")
add_custom_command(
  OUTPUT ${OUT}/residual_blocks.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/residual_blocks.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/residual_blocks.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/residual_blocks.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/residual_blocks.o
  DEPENDS residual_blocks.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/residual_blocks.c ${OUT}/residual_blocks.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS residual_blocks.c ${OUT}/residual_blocks.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_gqa_scores")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/gqa_scores")
add_custom_command(
  OUTPUT ${OUT}/gqa_scores.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/gqa_scores.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gqa_scores.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gqa_scores.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gqa_scores.o
  DEPENDS gqa_scores.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gqa_scores.c ${OUT}/gqa_scores.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS gqa_scores.c ${OUT}/gqa_scores.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_rmsnorm")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/rmsnorm")
add_custom_command(
  OUTPUT ${OUT}/rmsnorm.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rmsnorm.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rmsnorm.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rmsnorm.o
  DEPENDS rmsnorm.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm.c ${OUT}/rmsnorm.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rmsnorm.c ${OUT}/rmsnorm.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_attention")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/attention")
add_custom_command(
  OUTPUT ${OUT}/attention.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/attention.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/attention.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/attention.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/attention.o
  DEPENDS attention.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/attention.c ${OUT}/attention.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS attention.c ${OUT}/attention.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_attention_masked")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/attention_masked")
add_custom_command(
  OUTPUT ${OUT}/attention_masked.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/attention_masked.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/attention_masked.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/attention_masked.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/attention_masked.o
  DEPENDS attention_masked.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/attention_masked.c ${OUT}/attention_masked.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS attention_masked.c ${OUT}/attention_masked.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)

set(PROFILE dense)
set(HART ${DESIGN_HART_HART_${PROFILE}})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-${PROFILE}_attention_streamed")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/${PROFILE}/attention_streamed")
add_custom_command(
  OUTPUT ${OUT}/attention_streamed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${CMAKE_CURRENT_SOURCE_DIR}/attention_streamed.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/acc.mlir
    -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/banks.mlir
    -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=${PROFILE}" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_TARGET_OPT_${PROFILE}} ${OUT}/kernel.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -finalize-memref-to-llvm -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/attention_streamed.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/attention_streamed.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/attention_streamed.o
  DEPENDS attention_streamed.mlir ${DESIGN_TARGET_PLUGIN_${PROFILE}} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/attention_streamed.c ${OUT}/attention_streamed.o
    ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS attention_streamed.c ${OUT}/attention_streamed.o ${DESIGN_THREAD_BUILD_${PROFILE}}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(design-ctests-build ${TEST}-build)
