set(HART ${DESIGN_HART_HART_poly})
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_dynamic_mxint8")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/dynamic_mxint8")
add_custom_command(
  OUTPUT ${OUT}/dynamic_mxint8.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mxint8.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/dynamic_mxint8.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/dynamic_mxint8.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/dynamic_mxint8.o
  DEPENDS dynamic_mxint8.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mxint8.c ${OUT}/dynamic_mxint8.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS dynamic_mxint8.c ${OUT}/dynamic_mxint8.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

add_subdirectory(native)

# Explicit check for configurations with a Poly thread that has the activation Ball.
set(ACTIVATION_DESTINATION_OUT "${CMAKE_CURRENT_BINARY_DIR}/activation_destination")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/activation_destination.mlir")
add_custom_command(
  OUTPUT ${ACTIVATION_DESTINATION_OUT}/destination.checked
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ACTIVATION_DESTINATION_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/activation_destination.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly"
    -o ${ACTIVATION_DESTINATION_OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/activation_destination.mlir
    --check-prefix=HART --input-file=${ACTIVATION_DESTINATION_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${ACTIVATION_DESTINATION_OUT}/hart.mlir
    -canonicalize -convert-linalg-to-acc -o ${ACTIVATION_DESTINATION_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${ACTIVATION_DESTINATION_OUT}/attention.mlir
    -convert-linalg-to-acc -canonicalize -o ${ACTIVATION_DESTINATION_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ACTIVATION_DESTINATION_OUT}/acc.mlir
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -canonicalize -o ${ACTIVATION_DESTINATION_OUT}/buffers.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/activation_destination.mlir
    --check-prefix=BUFFER --input-file=${ACTIVATION_DESTINATION_OUT}/buffers.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${ACTIVATION_DESTINATION_OUT}/destination.checked
  DEPENDS activation_destination.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_target(${DESIGN_NAME}-hart${HART}-mlirtest-compiler_activation_destination-check
  DEPENDS ${ACTIVATION_DESTINATION_OUT}/destination.checked)
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/activation_destination.cpp")
add_custom_command(
  OUTPUT ${ACTIVATION_DESTINATION_OUT}/host.o
  COMMAND ${DESIGN_HART_OPT_poly} ${ACTIVATION_DESTINATION_OUT}/buffers.mlir
    -drop-equivalent-buffer-results=modify-public-functions
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -o ${ACTIVATION_DESTINATION_OUT}/buffers_out.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${ACTIVATION_DESTINATION_OUT}/buffers_out.mlir
    -assign-physical-banks -lower-acc -o ${ACTIVATION_DESTINATION_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${ACTIVATION_DESTINATION_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${ACTIVATION_DESTINATION_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ACTIVATION_DESTINATION_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${ACTIVATION_DESTINATION_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ACTIVATION_DESTINATION_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${ACTIVATION_DESTINATION_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ACTIVATION_DESTINATION_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${ACTIVATION_DESTINATION_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${ACTIVATION_DESTINATION_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${ACTIVATION_DESTINATION_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${ACTIVATION_DESTINATION_OUT}/host.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>"
    -S -o ${ACTIVATION_DESTINATION_OUT}/host_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${ACTIVATION_DESTINATION_OUT}/host_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${ACTIVATION_DESTINATION_OUT}/host.o
  DEPENDS ${ACTIVATION_DESTINATION_OUT}/destination.checked
  VERBATIM)
set(ACTIVATION_DESTINATION_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_activation_destination")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ACTIVATION_DESTINATION_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/activation_destination.cpp ${ACTIVATION_DESTINATION_OUT}/host.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${ACTIVATION_DESTINATION_TEST}-baremetal
  DEPENDS activation_destination.cpp ${ACTIVATION_DESTINATION_OUT}/host.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${ACTIVATION_DESTINATION_TEST}-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ACTIVATION_DESTINATION_TEST}-baremetal)
add_dependencies(poly-ctests-build ${ACTIVATION_DESTINATION_TEST}-build)

# Explicit target for the isolated attention thread with IntSoftmaxBall.
set(INTEGER_SOFTMAX_OUT "${CMAKE_CURRENT_BINARY_DIR}/integer_softmax")
set(INTEGER_SOFTMAX_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_integer_softmax")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.mlir")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.cpp")
add_custom_command(
  OUTPUT ${INTEGER_SOFTMAX_OUT}/host.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${INTEGER_SOFTMAX_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly"
    -o ${INTEGER_SOFTMAX_OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.mlir
    --check-prefix=HART --input-file=${INTEGER_SOFTMAX_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${INTEGER_SOFTMAX_OUT}/hart.mlir
    -convert-linalg-to-acc -o ${INTEGER_SOFTMAX_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${INTEGER_SOFTMAX_OUT}/attention.mlir
    -convert-linalg-to-acc -o ${INTEGER_SOFTMAX_OUT}/acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.mlir
    --check-prefix=ACC --input-file=${INTEGER_SOFTMAX_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${INTEGER_SOFTMAX_OUT}/acc.mlir
    -lower-frontend-fake-op -forward-materialized-tensors -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -canonicalize -cse -canonicalize -o ${INTEGER_SOFTMAX_OUT}/buffers.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.mlir
    --check-prefix=BUFFER --input-file=${INTEGER_SOFTMAX_OUT}/buffers.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${INTEGER_SOFTMAX_OUT}/buffers.mlir
    -drop-equivalent-buffer-results=modify-public-functions
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${INTEGER_SOFTMAX_OUT}/buffers_out.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${INTEGER_SOFTMAX_OUT}/buffers_out.mlir
    -assign-physical-banks -lower-acc -o ${INTEGER_SOFTMAX_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${INTEGER_SOFTMAX_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${INTEGER_SOFTMAX_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${INTEGER_SOFTMAX_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${INTEGER_SOFTMAX_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${INTEGER_SOFTMAX_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${INTEGER_SOFTMAX_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${INTEGER_SOFTMAX_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${INTEGER_SOFTMAX_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${INTEGER_SOFTMAX_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${INTEGER_SOFTMAX_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${INTEGER_SOFTMAX_OUT}/host.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>"
    -S -o ${INTEGER_SOFTMAX_OUT}/host_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${INTEGER_SOFTMAX_OUT}/host_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${INTEGER_SOFTMAX_OUT}/host.o
  DEPENDS integer_softmax.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${INTEGER_SOFTMAX_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/integer_softmax.cpp ${INTEGER_SOFTMAX_OUT}/host.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${INTEGER_SOFTMAX_TEST}-baremetal
  DEPENDS integer_softmax.cpp ${INTEGER_SOFTMAX_OUT}/host.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${INTEGER_SOFTMAX_TEST}-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${INTEGER_SOFTMAX_TEST}-baremetal)

set(MATRIX_WITH_BATCH_OUT "${CMAKE_CURRENT_BINARY_DIR}/matrix_with_batch")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/matrix_with_batch.mlir")
add_custom_command(
  OUTPUT ${MATRIX_WITH_BATCH_OUT}/partition.checked
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MATRIX_WITH_BATCH_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/matrix_with_batch.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${MATRIX_WITH_BATCH_OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_with_batch.mlir --input-file=${MATRIX_WITH_BATCH_OUT}/hart.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${MATRIX_WITH_BATCH_OUT}/partition.checked
  DEPENDS matrix_with_batch.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_target(${DESIGN_NAME}-hart${HART}-mlirtest-compiler_matrix_with_batch-check
  DEPENDS ${MATRIX_WITH_BATCH_OUT}/partition.checked)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-hart${HART}-mlirtest-compiler_matrix_with_batch-check)


set(M1_QUANT_CLOSURE_OUT "${CMAKE_CURRENT_BINARY_DIR}/m1_quant_closure")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/m1_quant_closure.mlir")
add_custom_command(
  OUTPUT ${M1_QUANT_CLOSURE_OUT}/partition.checked
  COMMAND ${CMAKE_COMMAND} -E make_directory ${M1_QUANT_CLOSURE_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/m1_quant_closure.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${M1_QUANT_CLOSURE_OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/m1_quant_closure.mlir --input-file=${M1_QUANT_CLOSURE_OUT}/hart.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${M1_QUANT_CLOSURE_OUT}/partition.checked
  DEPENDS m1_quant_closure.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_target(${DESIGN_NAME}-hart${HART}-mlirtest-compiler_m1_quant_closure-check
  DEPENDS ${M1_QUANT_CLOSURE_OUT}/partition.checked)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-hart${HART}-mlirtest-compiler_m1_quant_closure-check)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_mixed_attention")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/mixed_attention")
add_custom_command(
  OUTPUT ${OUT}/mixed_attention.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/mixed_attention.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/mixed_attention.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/mixed_attention.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/mixed_attention.o
  DEPENDS mixed_attention.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/mixed_attention.c ${OUT}/mixed_attention.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS mixed_attention.c ${OUT}/mixed_attention.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quantized_key")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/quantized_key")
add_custom_command(
  OUTPUT ${OUT}/quantized_key.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/quantized_key.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/quantized_key.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/quantized_key.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/quantized_key.o
  DEPENDS quantized_key.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized_key.c ${OUT}/quantized_key.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS quantized_key.c ${OUT}/quantized_key.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_dynamic_attention")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/dynamic_attention")
add_custom_command(
  OUTPUT ${OUT}/dynamic_attention.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_attention.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=compute,compute_transposed_rows,compute_transposed_tail_rows,compute_residual,compute_strided,compute_column_tiles,compute_compact,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/dynamic_attention.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/dynamic_attention.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/dynamic_attention.o
  DEPENDS dynamic_attention.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
# The residual rows run on the GP domain of the attention threads.
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/dynamic_attention.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_attention.c ${OUT}/dynamic_attention.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS dynamic_attention.c ${OUT}/dynamic_attention.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_packed_weights")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/packed_weights")
add_custom_command(
  OUTPUT ${OUT}/packed_weights.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/packed_weights.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/packed_weights.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/packed_weights.o
  DEPENDS packed_weights.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/packed_weights.c ${OUT}/packed_weights.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS packed_weights.c ${OUT}/packed_weights.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_mxint8_k_blocks")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/mxint8_k_blocks")
add_custom_command(
  OUTPUT ${OUT}/mxint8_k_blocks.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8_k_blocks.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/mxint8_k_blocks.mlir --check-prefix=BANK
    --input-file=${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/mxint8_k_blocks.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/mxint8_k_blocks.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/mxint8_k_blocks.o
  DEPENDS mxint8_k_blocks.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/stack/models/lib
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/mxint8_k_blocks.c ${OUT}/mxint8_k_blocks.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS mxint8_k_blocks.c ${OUT}/mxint8_k_blocks.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_offline_k_blocks")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/offline_k_blocks")
add_custom_command(
  OUTPUT ${OUT}/offline_k_blocks.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/offline_k_blocks.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/offline_k_blocks.mlir --check-prefix=PARTIAL
    --input-file=${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/offline_k_blocks.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/offline_k_blocks.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/offline_k_blocks.o
  DEPENDS offline_k_blocks.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/stack/models/lib
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/offline_k_blocks.c ${OUT}/offline_k_blocks.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS offline_k_blocks.c ${OUT}/offline_k_blocks.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(MLIR_TEST_PREFIX compiler)
set(MLIR_TEST_GROUP design-mlirtests-build)
add_mlir_test(reduce_sum
  TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_PROFILE}
  PASSES
    -convert-linalg-to-acc -canonicalize
    -one-shot-bufferize=bufferize-function-boundaries
    -drop-equivalent-buffer-results=modify-public-functions
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -assign-physical-banks -lower-acc -convert-linalg-to-loops
    -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-func-to-llvm -reconcile-unrealized-casts)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_direct_output_rows")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/direct_output_rows")
add_custom_command(
  OUTPUT ${OUT}/direct_output_rows.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/direct_output_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/direct_output_rows.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/direct_output_rows.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/direct_output_rows.o
  DEPENDS direct_output_rows.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    -DOUTPUT_READBACK_ROWS=${DESIGN_PROFILES_attention_THREAD_MEM_BANK_ENTRIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/direct_output_rows.c ${OUT}/direct_output_rows.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS direct_output_rows.c ${OUT}/direct_output_rows.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_head_view")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/head_view")
add_custom_command(
  OUTPUT ${OUT}/head_view.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/head_view.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/head_view.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/head_view.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/head_view.o
  DEPENDS head_view.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/head_view.c ${OUT}/head_view.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS head_view.c ${OUT}/head_view.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)


set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_compact_columns")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/compact_columns")
add_custom_command(
  OUTPUT ${OUT}/compact_columns.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/compact_columns.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/compact_columns.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/compact_columns.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/compact_columns.o
  DEPENDS compact_columns.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/compact_columns.c ${OUT}/compact_columns.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS compact_columns.c ${OUT}/compact_columns.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)


set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_compact_wide")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/compact_wide")
add_custom_command(
  OUTPUT ${OUT}/compact_wide.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/compact_wide.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/compact_wide.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/compact_wide.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/compact_wide.o
  DEPENDS compact_wide.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/compact_wide.c ${OUT}/compact_wide.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS compact_wide.c ${OUT}/compact_wide.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)


set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_compact_dynamic")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/compact_dynamic")
add_custom_command(
  OUTPUT ${OUT}/compact_dynamic.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/compact_dynamic.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/compact_dynamic.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/compact_dynamic.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/compact_dynamic.o
  DEPENDS compact_dynamic.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/compact_dynamic.c ${OUT}/compact_dynamic.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS compact_dynamic.c ${OUT}/compact_dynamic.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)


set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quantized_stride")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/quantized_stride")
add_custom_command(
  OUTPUT ${OUT}/quantized_stride.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/quantized_stride.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/quantized_stride.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/quantized_stride.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/quantized_stride.o
  DEPENDS quantized_stride.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/quantized_stride.c ${OUT}/quantized_stride.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS quantized_stride.c ${OUT}/quantized_stride.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_compact_lhs")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/compact_lhs")
add_custom_command(
  OUTPUT ${OUT}/compact_lhs.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/compact_lhs.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/compact_lhs.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/compact_lhs.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/compact_lhs.o
  DEPENDS compact_lhs.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/compact_lhs.c ${OUT}/compact_lhs.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS compact_lhs.c ${OUT}/compact_lhs.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(MLIR_TEST_PREFIX compiler)
set(MLIR_TEST_GROUP design-mlirtests-build)
add_mlir_test(reduce_quant_rows
  TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_PROFILE}
  PASSES
    -inline -convert-linalg-to-acc -canonicalize
    -one-shot-bufferize=bufferize-function-boundaries
    -drop-equivalent-buffer-results=modify-public-functions
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -assign-physical-banks -lower-acc -convert-linalg-to-loops
    -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-func-to-llvm -reconcile-unrealized-casts)

set(MLIR_TEST_PREFIX compiler)
set(MLIR_TEST_GROUP design-mlirtests-build)
add_mlir_test(reduce_quant_tail
  TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_PROFILE}
  PASSES
    -inline -convert-linalg-to-acc -canonicalize
    -one-shot-bufferize=bufferize-function-boundaries
    -drop-equivalent-buffer-results=modify-public-functions
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -assign-physical-banks -lower-acc -convert-linalg-to-loops
    -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-func-to-llvm -reconcile-unrealized-casts)

add_mlir_test(reduce_quant_short
  TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_PROFILE}
  PASSES
    -inline -convert-linalg-to-acc -canonicalize
    -one-shot-bufferize=bufferize-function-boundaries
    -drop-equivalent-buffer-results=modify-public-functions
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -assign-physical-banks -lower-acc -convert-linalg-to-loops
    -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-func-to-llvm -reconcile-unrealized-casts)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_gp_rows")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/gp_rows")
add_custom_command(
  OUTPUT ${OUT}/gp_rows.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/gp_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_rows.mlir
    --input-file=${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gp_rows.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gp_rows.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gp_rows.o
  DEPENDS gp_rows.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/gp_rows.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gp_rows.cpp ${OUT}/gp_rows.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS gp_rows.cpp ${OUT}/gp_rows.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_rotate_half")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/rotate_half")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/rotate_half.mlir")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/rotate_half.cpp")
add_custom_command(
  OUTPUT ${OUT}/rotate_half.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/rotate_half.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/rotate_half.mlir
    --check-prefix=HART --input-file=${OUT}/hart.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=compute_rotate_half,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -forward-materialized-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results=modify-public-functions
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/rotate_half.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/rotate_half.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/rotate_half.o
  DEPENDS rotate_half.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/rotate_half.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate_half.cpp ${OUT}/rotate_half.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS rotate_half.cpp ${OUT}/rotate_half.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(REDUCE_TEST_TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_PROFILE})
set(TEST "${DESIGN_NAME}-hart${DESIGN_WORKLOAD_CONTROL_HART}-mlirtest-compiler_gp_reduce")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/gp_reduce")
add_custom_command(
  OUTPUT ${OUT}/gp_reduce.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/gp_reduce.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=${DESIGN_PROFILES_${REDUCE_TEST_TARGET}_HART_COMPONENT}" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_reduce.mlir
    --input-file=${OUT}/acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,maximum,_binary_reduce_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/attention_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/addresses.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gp_reduce.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gp_reduce.ll
    -filetype=obj -mtriple=riscv64 -mattr=+xbuckyball,+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gp_reduce.o
  DEPENDS gp_reduce.mlir ${DESIGN_TARGET_PLUGIN_${REDUCE_TEST_TARGET}}
    ${DESIGN_PB}
  VERBATIM)
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_${REDUCE_TEST_TARGET}_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/reduce_operators.embed.o
  COMMAND ${DESIGN_TARGET_OPT_${REDUCE_TEST_TARGET}} ${OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_${REDUCE_TEST_TARGET}_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_${REDUCE_TEST_TARGET}_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/reduce_operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 reduce_operators.image reduce_operators.embed.o
  DEPENDS ${OUT}/gp_reduce.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gp_reduce.cpp ${OUT}/gp_reduce.o ${OUT}/reduce_operators.embed.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS gp_reduce.cpp ${OUT}/gp_reduce.o ${OUT}/reduce_operators.embed.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_gp_dynamic")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/gp_dynamic")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_dynamic.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_dynamic.mlir")
add_custom_command(
  OUTPUT ${OUT}/gp_dynamic.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/gp_dynamic.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_dynamic.mlir
    --check-prefix=POW --input-file=${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_dynamic.mlir
    --input-file=${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=compute,compute_pipeline,compute_strided,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gp_dynamic.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gp_dynamic.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gp_dynamic.o
  DEPENDS gp_dynamic.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/gp_dynamic.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gp_dynamic.cpp ${OUT}/gp_dynamic.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS gp_dynamic.cpp ${OUT}/gp_dynamic.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(POWER_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_broadcast_power")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/broadcast_power.c")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POWER_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_power.c ${OUT}/gp_dynamic.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${POWER_TEST}-baremetal
  DEPENDS broadcast_power.c ${OUT}/gp_dynamic.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${POWER_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POWER_TEST}-baremetal)
add_dependencies(poly-ctests-build ${POWER_TEST}-build)

set(SQUARE_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_square_sum")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/square_sum")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/square_sum.c")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/square_sum.mlir")
add_custom_command(
  OUTPUT ${OUT}/square_sum.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/square_sum.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/square_sum.mlir
    --input-file=${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=square_sum,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/square_sum.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/square_sum.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/square_sum.o
  DEPENDS square_sum.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR SQUARE_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${SQUARE_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/square_sum.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SQUARE_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/square_sum.c ${OUT}/square_sum.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${SQUARE_TEST}-baremetal
  DEPENDS square_sum.c ${OUT}/square_sum.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${SQUARE_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${SQUARE_TEST}-baremetal)
add_dependencies(poly-ctests-build ${SQUARE_TEST}-build)

set(TANH_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_tanh")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/tanh")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/tanh.c")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/tanh.mlir")
add_custom_command(
  OUTPUT ${OUT}/tanh.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/tanh.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/tanh.mlir
    --input-file=${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=tanh_rows,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/tanh.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/tanh.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/tanh.o
  DEPENDS tanh.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR TANH_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${TANH_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/tanh.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TANH_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/tanh.c ${OUT}/tanh.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TANH_TEST}-baremetal
  DEPENDS tanh.c ${OUT}/tanh.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TANH_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TANH_TEST}-baremetal)
add_dependencies(poly-ctests-build ${TANH_TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_gp_mixed")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/gp_mixed")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_mixed.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_mixed.mlir")
add_custom_command(
  OUTPUT ${OUT}/gp_mixed.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/gp_mixed.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_mixed.mlir
    --input-file=${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=compute_mixed,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_mixed.mlir
    --check-prefix=BATCH --input-file=${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gp_mixed.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/gp_mixed.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>"
    -S -o ${OUT}/gp_mixed_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gp_mixed_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gp_mixed.o
  DEPENDS gp_mixed.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/gp_mixed.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gp_mixed.cpp ${OUT}/gp_mixed.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS gp_mixed.cpp ${OUT}/gp_mixed.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_gp_decode")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/gp_decode")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_decode.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_decode.mlir")
add_custom_command(
  OUTPUT ${OUT}/gp_decode.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/gp_decode.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_decode.mlir
    --input-file=${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_decode.mlir
    --check-prefix=DECODE --input-file=${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=compute_decode,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gp_decode.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/gp_decode.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>"
    -S -o ${OUT}/gp_decode_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gp_decode_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gp_decode.o
  DEPENDS gp_decode.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
math(EXPR GP_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${GP_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/gp_decode.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gp_decode.cpp ${OUT}/gp_decode.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS gp_decode.cpp ${OUT}/gp_decode.o ${OUT}/operators.embed.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(COLUMNS_OUT "${CMAKE_CURRENT_BINARY_DIR}/matrix_columns")
set(COLUMNS_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_matrix_columns")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.mlir")
add_custom_command(
  OUTPUT ${COLUMNS_OUT}/matrix_columns.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${COLUMNS_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${COLUMNS_OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.mlir
    --input-file=${COLUMNS_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${COLUMNS_OUT}/hart.mlir
    -convert-linalg-to-acc -o ${COLUMNS_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${COLUMNS_OUT}/attention.mlir
    -convert-linalg-to-acc -o ${COLUMNS_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${COLUMNS_OUT}/acc.mlir
    "-symbol-privatize=exclude=compute_columns" -symbol-dce -lower-frontend-fake-op
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${COLUMNS_OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${COLUMNS_OUT}/bufferized.mlir
    -assign-physical-banks -lower-acc -o ${COLUMNS_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${COLUMNS_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${COLUMNS_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${COLUMNS_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${COLUMNS_OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.mlir
    --check-prefix=BATCH --implicit-check-not=memref.copy --implicit-check-not=linalg.copy
    "--implicit-check-not=call @stream_copy" --input-file=${COLUMNS_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${COLUMNS_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${COLUMNS_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${COLUMNS_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${COLUMNS_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${COLUMNS_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${COLUMNS_OUT}/matrix_columns.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${COLUMNS_OUT}/matrix_columns.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>"
    -S -o ${COLUMNS_OUT}/matrix_columns_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${COLUMNS_OUT}/matrix_columns_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${COLUMNS_OUT}/matrix_columns.o
  DEPENDS matrix_columns.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${COLUMNS_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20 -DCOLUMN_K=512 -DCOLUMN_CHUNKS=1 -DCOLUMN_ROWS=1
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.cpp ${COLUMNS_OUT}/matrix_columns.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${COLUMNS_TEST}-baremetal
  DEPENDS matrix_columns.cpp ${COLUMNS_OUT}/matrix_columns.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${COLUMNS_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${COLUMNS_TEST}-baremetal)
add_dependencies(poly-ctests-build ${COLUMNS_TEST}-build)

set(ROWS_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_matrix_columns_rows")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ROWS_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20 -DCOLUMN_K=512 -DCOLUMN_CHUNKS=1 -DCOLUMN_ROWS=5
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.cpp ${COLUMNS_OUT}/matrix_columns.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${ROWS_TEST}-baremetal
  DEPENDS matrix_columns.cpp ${COLUMNS_OUT}/matrix_columns.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${ROWS_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ROWS_TEST}-baremetal)
add_dependencies(poly-ctests-build ${ROWS_TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_matrix_columns_blocked")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/matrix_columns_blocked")
add_custom_command(
  OUTPUT ${OUT}/matrix_columns_blocked.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns_blocked.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -lower-frontend-fake-op -inline -eliminate-empty-tensors
    -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir --buddy-to-llvmir -o ${OUT}/matrix.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/matrix.ll -mtriple=riscv64
    -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${OUT}/matrix_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/matrix_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/matrix_columns_blocked.o
  DEPENDS matrix_columns_blocked.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20 -fconstexpr-ops-limit=100000000
    -DCOLUMN_K=2048 -DCOLUMN_CHUNKS=2 -DCOLUMN_ROWS=5
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix_columns.cpp ${OUT}/matrix_columns_blocked.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS matrix_columns.cpp ${OUT}/matrix_columns_blocked.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(PERMUTE_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_gp_permute")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/gp_permute")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_permute.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/gp_permute.mlir")
add_custom_command(
  OUTPUT ${OUT}/gp_permute.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/gp_permute.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gp_permute.mlir
    --input-file=${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=permute_rows,_binary_operators_image_start" -symbol-dce -o ${OUT}/host.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/host.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -inline -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/gp_permute.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/gp_permute.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/gp_permute.o
  DEPENDS gp_permute.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR PERMUTE_CODE_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata
    -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm
    -convert-arith-to-llvm -finalize-memref-to-llvm=use-aligned-alloc
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${PERMUTE_CODE_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${OUT}/operators.o -o ${OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${OUT}/operators.elf ${OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${OUT}/gp_permute.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PERMUTE_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/gp_permute.cpp ${OUT}/gp_permute.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${PERMUTE_TEST}-baremetal
  DEPENDS gp_permute.cpp ${OUT}/gp_permute.o ${OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${PERMUTE_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PERMUTE_TEST}-baremetal)
add_dependencies(poly-ctests-build ${PERMUTE_TEST}-build)



set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_prefetch_weights")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/prefetch_weights")
add_custom_command(
  OUTPUT ${OUT}/prefetch_weights.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/prefetch_weights.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/prefetch_weights.mlir --check-prefix=REUSE
    --input-file=${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/prefetch_weights.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/prefetch_weights.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/prefetch_weights.o
  DEPENDS prefetch_weights.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    ${CMAKE_CURRENT_SOURCE_DIR}/prefetch_weights.c ${OUT}/prefetch_weights.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS prefetch_weights.c ${OUT}/prefetch_weights.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_packed_input")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/packed_input")
add_custom_command(
  OUTPUT ${OUT}/packed_input.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/packed_input.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/packed_input.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/packed_input.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/packed_input.o
  DEPENDS packed_input.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    ${CMAKE_CURRENT_SOURCE_DIR}/packed_input.c ${OUT}/packed_input.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS packed_input.c ${OUT}/packed_input.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_resident_rows")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/resident_rows")
add_custom_command(
  OUTPUT ${OUT}/resident_rows.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/resident_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/resident_rows.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/resident_rows.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/resident_rows.o
  DEPENDS resident_rows.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    ${CMAKE_CURRENT_SOURCE_DIR}/resident_rows.c ${OUT}/resident_rows.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS resident_rows.c ${OUT}/resident_rows.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)
set(HETERO_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_heterogeneous")
set(HETERO_OUT "${CMAKE_CURRENT_BINARY_DIR}/heterogeneous")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous.mlir")
add_custom_command(
  OUTPUT ${HETERO_OUT}/host.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${HETERO_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${HETERO_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${HETERO_OUT}/hart.mlir
    -convert-linalg-to-acc -o ${HETERO_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${HETERO_OUT}/attention.mlir
    -convert-linalg-to-acc -o ${HETERO_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${HETERO_OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce
    -lower-frontend-fake-op -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${HETERO_OUT}/buffers.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${HETERO_OUT}/buffers.mlir
    -assign-physical-banks -lower-acc -o ${HETERO_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${HETERO_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${HETERO_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${HETERO_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${HETERO_OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous.mlir
    --check-prefix=SHM --input-file=${HETERO_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${HETERO_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${HETERO_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${HETERO_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${HETERO_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${HETERO_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${HETERO_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${HETERO_OUT}/host.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${HETERO_OUT}/host_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${HETERO_OUT}/host_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${HETERO_OUT}/host.o
  DEPENDS heterogeneous.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR HETERO_GP_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${HETERO_OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${HETERO_OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${HETERO_OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${HETERO_OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm -convert-arith-to-llvm
    -finalize-memref-to-llvm=use-aligned-alloc -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${HETERO_OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${HETERO_OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${HETERO_OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${HETERO_OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${HETERO_OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${HETERO_OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${HETERO_OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${HETERO_GP_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${HETERO_OUT}/operators.o -o ${HETERO_OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${HETERO_OUT}/operators.elf ${HETERO_OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${HETERO_OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${HETERO_OUT}/host.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HETERO_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/heterogeneous.cpp ${HETERO_OUT}/host.o ${HETERO_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${HETERO_TEST}-baremetal
  DEPENDS heterogeneous.cpp ${HETERO_OUT}/host.o ${HETERO_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${HETERO_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${HETERO_TEST}-baremetal)
add_dependencies(poly-ctests-build ${HETERO_TEST}-build)

set(CHUNKED_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_chunked_rows")
set(CHUNKED_OUT "${CMAKE_CURRENT_BINARY_DIR}/chunked_rows")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/chunked_rows.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/chunked_rows.mlir")
add_custom_command(
  OUTPUT ${CHUNKED_OUT}/host.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CHUNKED_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/chunked_rows.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${CHUNKED_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${CHUNKED_OUT}/hart.mlir
    -convert-linalg-to-acc -o ${CHUNKED_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${CHUNKED_OUT}/attention.mlir
    -convert-linalg-to-acc -o ${CHUNKED_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${CHUNKED_OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce
    -lower-frontend-fake-op -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${CHUNKED_OUT}/buffers.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${CHUNKED_OUT}/buffers.mlir
    -assign-physical-banks -lower-acc -o ${CHUNKED_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${CHUNKED_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${CHUNKED_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${CHUNKED_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${CHUNKED_OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/chunked_rows.mlir
    --check-prefix=SHM --input-file=${CHUNKED_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${CHUNKED_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${CHUNKED_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${CHUNKED_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${CHUNKED_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${CHUNKED_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${CHUNKED_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${CHUNKED_OUT}/host.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${CHUNKED_OUT}/host_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${CHUNKED_OUT}/host_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${CHUNKED_OUT}/host.o
  DEPENDS chunked_rows.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR CHUNKED_GP_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${CHUNKED_OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${CHUNKED_OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${CHUNKED_OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${CHUNKED_OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm -convert-arith-to-llvm
    -finalize-memref-to-llvm=use-aligned-alloc -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${CHUNKED_OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${CHUNKED_OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${CHUNKED_OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${CHUNKED_OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${CHUNKED_OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${CHUNKED_OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${CHUNKED_OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${CHUNKED_GP_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${CHUNKED_OUT}/operators.o -o ${CHUNKED_OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${CHUNKED_OUT}/operators.elf ${CHUNKED_OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${CHUNKED_OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${CHUNKED_OUT}/host.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CHUNKED_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/chunked_rows.cpp ${CHUNKED_OUT}/host.o ${CHUNKED_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${CHUNKED_TEST}-baremetal
  DEPENDS chunked_rows.cpp ${CHUNKED_OUT}/host.o ${CHUNKED_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${CHUNKED_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CHUNKED_TEST}-baremetal)
add_dependencies(poly-ctests-build ${CHUNKED_TEST}-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/packed_short.c")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/packed_short.mlir")
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_packed_short")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/packed_short")
add_custom_command(
  OUTPUT ${OUT}/packed_short.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/packed_short.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/packed_short.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/packed_short.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/packed_short.o
  DEPENDS packed_short.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    ${CMAKE_CURRENT_SOURCE_DIR}/packed_short.c ${OUT}/packed_short.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS packed_short.c ${OUT}/packed_short.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

set(ROW352_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quant_row352")
set(ROW352_OUT "${CMAKE_CURRENT_BINARY_DIR}/quant_row352")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_row352.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_row352.mlir")
add_custom_command(
  OUTPUT ${ROW352_OUT}/host.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ROW352_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/quant_row352.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${ROW352_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${ROW352_OUT}/hart.mlir
    -convert-linalg-to-acc -o ${ROW352_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${ROW352_OUT}/attention.mlir
    -convert-linalg-to-acc -o ${ROW352_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW352_OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce
    -lower-frontend-fake-op -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${ROW352_OUT}/buffers.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${ROW352_OUT}/buffers.mlir
    -assign-physical-banks -lower-acc -o ${ROW352_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${ROW352_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${ROW352_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW352_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${ROW352_OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/quant_row352.mlir
    --check-prefix=SHM --input-file=${ROW352_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW352_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${ROW352_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW352_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${ROW352_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${ROW352_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${ROW352_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${ROW352_OUT}/host.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${ROW352_OUT}/host_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${ROW352_OUT}/host_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${ROW352_OUT}/host.o
  DEPENDS quant_row352.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR ROW352_GP_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${ROW352_OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW352_OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${ROW352_OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${ROW352_OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm -convert-arith-to-llvm
    -finalize-memref-to-llvm=use-aligned-alloc -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${ROW352_OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${ROW352_OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${ROW352_OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${ROW352_OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${ROW352_OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${ROW352_OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${ROW352_OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${ROW352_GP_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${ROW352_OUT}/operators.o -o ${ROW352_OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${ROW352_OUT}/operators.elf ${ROW352_OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${ROW352_OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${ROW352_OUT}/host.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ROW352_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_row352.cpp ${ROW352_OUT}/host.o ${ROW352_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${ROW352_TEST}-baremetal
  DEPENDS quant_row352.cpp ${ROW352_OUT}/host.o ${ROW352_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${ROW352_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ROW352_TEST}-baremetal)
add_dependencies(poly-ctests-build ${ROW352_TEST}-build)

set(ROW544_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quant_row544")
set(ROW544_OUT "${CMAKE_CURRENT_BINARY_DIR}/quant_row544")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_row544.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_row544.mlir")
add_custom_command(
  OUTPUT ${ROW544_OUT}/host.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ROW544_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/quant_row544.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${ROW544_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${ROW544_OUT}/hart.mlir
    -convert-linalg-to-acc -o ${ROW544_OUT}/attention.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${ROW544_OUT}/attention.mlir
    -convert-linalg-to-acc -o ${ROW544_OUT}/acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW544_OUT}/acc.mlir
    "-symbol-privatize=exclude=compute,_binary_operators_image_start" -symbol-dce
    -lower-frontend-fake-op -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${ROW544_OUT}/buffers.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${ROW544_OUT}/buffers.mlir
    -assign-physical-banks -lower-acc -o ${ROW544_OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${ROW544_OUT}/attention_instructions.mlir
    -assign-physical-banks -lower-acc -o ${ROW544_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW544_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${ROW544_OUT}/kernel.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/quant_row544.mlir
    --check-prefix=SHM --input-file=${ROW544_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW544_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${ROW544_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW544_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${ROW544_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${ROW544_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${ROW544_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${ROW544_OUT}/host.ll
    -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${ROW544_OUT}/host_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${ROW544_OUT}/host_opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${ROW544_OUT}/host.o
  DEPENDS quant_row544.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
math(EXPR ROW544_GP_BYTES "${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_I_BUF_WORDS} * 4")
add_custom_command(
  OUTPUT ${ROW544_OUT}/operators.embed.o
  COMMAND ${DESIGN_HART_OPT_poly} ${ROW544_OUT}/acc.mlir
    "-symbol-privatize=exclude=operator,gp_space" -symbol-dce -o ${ROW544_OUT}/operators.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt ${ROW544_OUT}/operators.mlir
    -convert-vector-to-scf -convert-vector-to-llvm -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -convert-cf-to-llvm -convert-math-to-llvm -convert-arith-to-llvm
    -finalize-memref-to-llvm=use-aligned-alloc -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${ROW544_OUT}/operators_llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${ROW544_OUT}/operators_llvm.mlir
    --buddy-to-llvmir -o ${ROW544_OUT}/operators.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${ROW544_OUT}/operators.ll
    -mtriple=riscv64 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR} "-passes=default<O2>"
    -S -o ${ROW544_OUT}/operators_opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${ROW544_OUT}/operators_opt.ll
    -filetype=obj -mtriple=riscv64 -O2 -mattr=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_LLVM_MATTR}
    -riscv-v-vector-bits-min=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -riscv-v-vector-bits-max=${DESIGN_PROFILES_attention_THREAD_GPDOMAIN_V_LEN}
    -target-abi=lp64f -o ${ROW544_OUT}/operators.o
  COMMAND ${ELF_CC} -nostdlib -nostartfiles -march=rv64imafd -mabi=lp64f
    -Wl,--defsym=operator_text_bytes=${ROW544_GP_BYTES}
    -Wl,-T,${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
    ${ROW544_OUT}/operators.o -o ${ROW544_OUT}/operators.elf
  COMMAND ${ELF_OBJCOPY} -O binary -j .header -j .text -j .rodata
    ${ROW544_OUT}/operators.elf ${ROW544_OUT}/operators.image
  COMMAND ${CMAKE_COMMAND} -E chdir ${ROW544_OUT} ${ELF_OBJCOPY}
    -I binary -O elf64-littleriscv -B riscv
    --rename-section .data=.rodata,alloc,load,readonly,data,contents
    --set-section-alignment .data=16 operators.image operators.embed.o
  DEPENDS ${ROW544_OUT}/host.o ${DESIGN_REPO}/examples/designs/poly/harts/poly/stack/operators.ld
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ROW544_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_row544.cpp ${ROW544_OUT}/host.o ${ROW544_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${ROW544_TEST}-baremetal
  DEPENDS quant_row544.cpp ${ROW544_OUT}/host.o ${ROW544_OUT}/operators.embed.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${ROW544_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${ROW544_TEST}-baremetal)
add_dependencies(poly-ctests-build ${ROW544_TEST}-build)


ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_split.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_split.mlir")
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quant_split")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/quant_split")
add_custom_command(
  OUTPUT ${OUT}/quant_split.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/quant_split.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/quant_split.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/quant_split.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/quant_split.o
  DEPENDS quant_split.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_split.cpp ${OUT}/quant_split.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS quant_split.cpp ${OUT}/quant_split.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_planes.cpp")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/quant_planes.mlir")
set(TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_quant_planes")
set(OUT "${CMAKE_CURRENT_BINARY_DIR}/quant_planes")
add_custom_command(
  OUTPUT ${OUT}/quant_planes.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/quant_planes.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/hart.mlir
    -convert-linalg-to-acc -o ${OUT}/attention_acc.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_acc.mlir
    -convert-linalg-to-acc -o ${OUT}/ffn_acc.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_acc.mlir -lower-frontend-fake-op -o ${OUT}/frontend.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/frontend.mlir
    -eliminate-empty-tensors -convert-elementwise-to-linalg
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -o ${OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/bufferized.mlir
    -assign-physical-banks -o ${OUT}/attention_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${OUT}/attention_banks.mlir
    -lower-acc -o ${OUT}/attention_instructions.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/attention_instructions.mlir
    -assign-physical-banks -o ${OUT}/ffn_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${OUT}/ffn_banks.mlir
    -lower-acc -o ${OUT}/ffn_instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/ffn_instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize -o ${OUT}/addresses.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/addresses.mlir
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))" -o ${OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-arith-to-llvm
    -convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
    -reconcile-unrealized-casts -o ${OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${OUT}/llvm.mlir
    --buddy-to-llvmir -o ${OUT}/quant_planes.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${OUT}/quant_planes.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${OUT}/quant_planes.o
  DEPENDS quant_planes.mlir ${DESIGN_HART_PLUGIN_poly} ${DESIGN_PB}
  VERBATIM)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_planes.cpp ${OUT}/quant_planes.o
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal
  DEPENDS quant_planes.cpp ${OUT}/quant_planes.o ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST}-baremetal)
add_dependencies(poly-ctests-build ${TEST}-build)
