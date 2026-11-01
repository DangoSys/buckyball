set(MLIR_TEST_GROUP linalg-mlir-tests-build)
set(MLIR_TEST_PREFIX linalg)

add_custom_target(linalg-mlir-tests-build)

add_mlir_test(conv2d_nhwc_hwcf
  TARGET toy
  PASSES
    -convert-linalg-to-acc
    -lower-frontend-fake-op
    -eliminate-empty-tensors
    -one-shot-bufferize=bufferize-function-boundaries
    -ownership-based-buffer-deallocation
    -buffer-deallocation-simplification
    -bufferization-lower-deallocations
    -assign-physical-banks
    -lower-acc
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

add_mlir_test(conv2d_nhwc_hwcf_deep
  TARGET toy
  PASSES
    -convert-linalg-to-acc
    -lower-frontend-fake-op
    -eliminate-empty-tensors
    -one-shot-bufferize=bufferize-function-boundaries
    -ownership-based-buffer-deallocation
    -buffer-deallocation-simplification
    -bufferization-lower-deallocations
    -assign-physical-banks
    -lower-acc
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

add_mlir_test(conv2d_nhwc_fhwc
  TARGET toy
  PASSES
    -convert-linalg-to-acc
    -lower-frontend-fake-op
    -eliminate-empty-tensors
    -one-shot-bufferize=bufferize-function-boundaries
    -ownership-based-buffer-deallocation
    -buffer-deallocation-simplification
    -bufferization-lower-deallocations
    -assign-physical-banks
    -lower-acc
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

add_mlir_test(conv2d_nchw_fchw
  TARGET toy
  PASSES
    -convert-linalg-to-acc
    -lower-frontend-fake-op
    -eliminate-empty-tensors
    -one-shot-bufferize=bufferize-function-boundaries
    -ownership-based-buffer-deallocation
    -buffer-deallocation-simplification
    -bufferization-lower-deallocations
    -assign-physical-banks
    -lower-acc
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
