add_custom_target(pebble-mlir-tests-build)

set(PEBBLE_CUSTOM_PASSES
  -allow-unregistered-dialect
  -form-custom-ops
  -convert-linalg-to-buckyball
  -lower-frontend
  -eliminate-empty-tensors
  -one-shot-bufferize=bufferize-function-boundaries
  -ownership-based-buffer-deallocation
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -convert-trace-to-llvm
  -assign-physical-banks
  -lower-buckyball
  -convert-linalg-to-loops
  -expand-strided-metadata
  -lower-affine
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-func-to-llvm
  -reconcile-unrealized-casts)

add_subdirectory(bank)
add_subdirectory(core)
add_subdirectory(linalg)

add_dependencies(design-mlirtests-build pebble-mlir-tests-build)
