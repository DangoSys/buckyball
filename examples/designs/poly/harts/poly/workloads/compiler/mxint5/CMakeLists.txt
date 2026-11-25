set(HART ${DESIGN_HART_HART_poly})

# Five-column MXINT5 closures use the two targets explicitly present here.
set(MX5_COLUMNS_OUT "${CMAKE_CURRENT_BINARY_DIR}/mxint5_columns")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/mxint5_columns.mlir")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/mxint5_columns.cpp")
add_custom_command(
  OUTPUT ${MX5_COLUMNS_OUT}/host.o
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MX5_COLUMNS_OUT}
  COMMAND ${DESIGN_HART_OPT_poly} ${CMAKE_CURRENT_SOURCE_DIR}/mxint5_columns.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=poly" -o ${MX5_COLUMNS_OUT}/hart.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/mxint5_columns.mlir
    --check-prefix=HART --input-file=${MX5_COLUMNS_OUT}/hart.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${MX5_COLUMNS_OUT}/hart.mlir
    -convert-linalg-to-mixed5 -canonicalize -convert-linalg-to-acc -o ${MX5_COLUMNS_OUT}/mixed5.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${MX5_COLUMNS_OUT}/mixed5.mlir
    -convert-linalg-to-ffn5 -canonicalize -convert-linalg-to-acc -o ${MX5_COLUMNS_OUT}/ffn5.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${MX5_COLUMNS_OUT}/ffn5.mlir
    -lower-frontend-fake-op -inline -eliminate-empty-tensors
    -convert-elementwise-to-linalg -canonicalize -o ${MX5_COLUMNS_OUT}/inlined.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${MX5_COLUMNS_OUT}/inlined.mlir
    "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=infer-layout-map"
    -drop-equivalent-buffer-results
    "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize -cse -canonicalize
    -o ${MX5_COLUMNS_OUT}/bufferized.mlir
  COMMAND ${DESIGN_TARGET_OPT_attention} ${MX5_COLUMNS_OUT}/bufferized.mlir
    -assign-physical-banks -lower-acc -o ${MX5_COLUMNS_OUT}/mixed5_banks.mlir
  COMMAND ${DESIGN_TARGET_OPT_ffn} ${MX5_COLUMNS_OUT}/mixed5_banks.mlir
    -assign-physical-banks -lower-acc -o ${MX5_COLUMNS_OUT}/instructions.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${MX5_COLUMNS_OUT}/instructions.mlir
    -expand-strided-metadata -lower-affine -canonicalize
    "-lower-hart=design=${DESIGN_PB} hart=poly" -o ${MX5_COLUMNS_OUT}/kernel.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${MX5_COLUMNS_OUT}/kernel.mlir
    -pass-pipeline "builtin.module(func.func(convert-strided-memref-copy-to-linalg))"
    -o ${MX5_COLUMNS_OUT}/copies.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/mxint5_columns.mlir
    --check-prefix=COPIES --input-file=${MX5_COLUMNS_OUT}/copies.mlir
  COMMAND ${DESIGN_HART_OPT_poly} ${MX5_COLUMNS_OUT}/copies.mlir
    -convert-linalg-to-loops -expand-strided-metadata -lower-affine
    -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm
    -convert-arith-to-llvm -convert-math-to-llvm -convert-math-to-libm
    -convert-func-to-llvm -reconcile-unrealized-casts -o ${MX5_COLUMNS_OUT}/llvm.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-translate ${MX5_COLUMNS_OUT}/llvm.mlir
    --buddy-to-llvmir -o ${MX5_COLUMNS_OUT}/host.ll
  COMMAND ${DESIGN_MLIR}/../../../bin/opt ${MX5_COLUMNS_OUT}/host.ll -mtriple=riscv64
    -mattr=+m,+a,+f,+d,+c "-passes=default<O2>" -S -o ${MX5_COLUMNS_OUT}/host-opt.ll
  COMMAND ${DESIGN_COMPILER}/bin/buddy-llc ${MX5_COLUMNS_OUT}/host-opt.ll
    -filetype=obj -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -target-abi=lp64d
    -code-model=medium -O2 -o ${MX5_COLUMNS_OUT}/host.o
  DEPENDS mxint5_columns.mlir ${DESIGN_HART_PLUGIN_poly}
    ${DESIGN_TARGET_PLUGIN_attention} ${DESIGN_TARGET_PLUGIN_ffn} ${DESIGN_PB}
  VERBATIM)
foreach(ROWS IN ITEMS 1 3)
  set(MX5_COLUMNS_TEST "${DESIGN_NAME}-hart${HART}-mlirtest-compiler_mxint5_columns_m${ROWS}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MX5_COLUMNS_TEST}-baremetal
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS} -std=c++20
      -DOUTPUT_CHANNELS=160 -DINPUT_K=1024 -DINPUT_ROWS=${ROWS}
      -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
      -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
      $<TARGET_PROPERTY:bbsw-crt0-hart${HART}-baremetal,INTERFACE_LINK_OPTIONS>
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
      ${CMAKE_CURRENT_SOURCE_DIR}/mxint5_columns.cpp ${MX5_COLUMNS_OUT}/host.o
      ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
      $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
      -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${MX5_COLUMNS_TEST}-baremetal
    DEPENDS mxint5_columns.cpp ${MX5_COLUMNS_OUT}/host.o
      ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
      runtime-baremetal kernel
      $<TARGET_OBJECTS:bbsw-crt0-hart${HART}-baremetal>
      $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(${MX5_COLUMNS_TEST}-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MX5_COLUMNS_TEST}-baremetal)
endforeach()
