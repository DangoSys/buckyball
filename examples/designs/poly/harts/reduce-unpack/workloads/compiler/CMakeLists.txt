set(MLIR_TEST_PREFIX compiler)
set(MLIR_TEST_GROUP design-mlirtests-build)
add_mlir_test(integer_norm TARGET main
  PASSES
    -inline -convert-linalg-to-acc -canonicalize -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries"
    -drop-equivalent-buffer-results
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -assign-physical-banks -lower-acc -convert-linalg-to-loops
    -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm
    -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts
  LIBRARIES bbsw-thread-main-linux
  BAREMETAL_LIBRARIES bbsw-thread-main-baremetal)

add_mlir_test(packed_decode TARGET main
  PASSES
    -inline -convert-linalg-to-acc -canonicalize -eliminate-empty-tensors
    "-one-shot-bufferize=bufferize-function-boundaries"
    -drop-equivalent-buffer-results
    -ownership-based-buffer-deallocation -buffer-deallocation-simplification
    -bufferization-lower-deallocations -canonicalize
    -assign-physical-banks -lower-acc -convert-linalg-to-loops
    -expand-strided-metadata -lower-affine -convert-scf-to-cf
    -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm
    -finalize-memref-to-llvm -convert-func-to-llvm
    -reconcile-unrealized-casts
  LIBRARIES bbsw-thread-main-linux
  BAREMETAL_LIBRARIES bbsw-thread-main-baremetal)

set(PACKED_SUM_OUT "${CMAKE_CURRENT_BINARY_DIR}/packed_sum")
add_custom_command(
  OUTPUT ${PACKED_SUM_OUT}/acc.mlir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PACKED_SUM_OUT}
  COMMAND ${DESIGN_TARGET_OPT_main} ${CMAKE_CURRENT_SOURCE_DIR}/packed_sum.mlir
    "-convert-linalg-to-hart=design=${DESIGN_PB} hart=reduce-unpack"
    -convert-linalg-to-acc -o ${PACKED_SUM_OUT}/acc.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
    ${CMAKE_CURRENT_SOURCE_DIR}/packed_sum.mlir --check-prefix=FUSED
    --input-file=${PACKED_SUM_OUT}/acc.mlir
  DEPENDS packed_sum.mlir ${DESIGN_TARGET_PLUGIN_main} ${DESIGN_PB}
  VERBATIM)
add_custom_target(poly-reduce-unpack-packed-sum-build
  DEPENDS ${PACKED_SUM_OUT}/acc.mlir)
add_dependencies(design-mlirtests-build poly-reduce-unpack-packed-sum-build)
