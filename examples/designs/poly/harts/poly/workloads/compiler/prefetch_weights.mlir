// REUSE-LABEL: func.func private @pipeline(
// REUSE: call @stream_begin
// REUSE: call @pipeline_weights_launch
// REUSE-COUNT-10: call @stream_transfer
// REUSE-LABEL: func.func private @pipeline_weights(
// REUSE-NOT: memref.copy
// REUSE-NOT: call @stream_allocate
// REUSE: return
func.func private @pipeline(%a: tensor<?x48xi8>, %w0: tensor<1536xi8>, %w1: tensor<1584xi8>, %out: tensor<?x32xf32>) -> tensor<?x32xf32> {
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %a, %c0 : tensor<?x48xi8>
  %zero = arith.constant 0.0 : f32
  %a0 = "frontend.mxint8_to_fp32"(%a) {axis = 1 : i64} : (tensor<?x48xi8>) -> tensor<?x32xf32>
  %p0 = tensor.expand_shape %w0 [[0, 1]] output_shape [32, 48] : tensor<1536xi8> into tensor<32x48xi8>
  %d0 = "frontend.mxint8_to_fp32"(%p0) {axis = 1 : i64} : (tensor<32x48xi8>) -> tensor<32x32xf32>
  %e0 = tensor.empty() : tensor<32x32xf32>
  %t0 = linalg.transpose ins(%d0 : tensor<32x32xf32>) outs(%e0 : tensor<32x32xf32>) permutation = [1, 0]
  %e1 = tensor.empty(%rows) : tensor<?x32xf32>
  %z0 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<?x32xf32>) -> tensor<?x32xf32>
  %r0 = linalg.matmul ins(%a0, %t0 : tensor<?x32xf32>, tensor<32x32xf32>) outs(%z0 : tensor<?x32xf32>) -> tensor<?x32xf32>
  %q = "frontend.fp32_to_mxint8"(%r0) {axis = 1 : i64} : (tensor<?x32xf32>) -> tensor<?x33xi8>
  %a1 = "frontend.mxint8_to_fp32"(%q) {axis = 1 : i64} : (tensor<?x33xi8>) -> tensor<?x32xf32>
  %s1 = tensor.extract_slice %w1[0] [1536] [1] : tensor<1584xi8> to tensor<1536xi8>
  %p1 = tensor.expand_shape %s1 [[0, 1]] output_shape [32, 48] : tensor<1536xi8> into tensor<32x48xi8>
  %d1 = "frontend.mxint8_to_fp32"(%p1) {axis = 1 : i64} : (tensor<32x48xi8>) -> tensor<32x32xf32>
  %e2 = tensor.empty() : tensor<32x32xf32>
  %t1 = linalg.transpose ins(%d1 : tensor<32x32xf32>) outs(%e2 : tensor<32x32xf32>) permutation = [1, 0]
  %e3 = tensor.empty(%rows) : tensor<?x32xf32>
  %z1 = linalg.fill ins(%zero : f32) outs(%e3 : tensor<?x32xf32>) -> tensor<?x32xf32>
  %r1 = linalg.matmul ins(%a1, %t1 : tensor<?x32xf32>, tensor<32x32xf32>) outs(%z1 : tensor<?x32xf32>) -> tensor<?x32xf32>
  %written = bufferization.materialize_in_destination %r1 in %out : (tensor<?x32xf32>, tensor<?x32xf32>) -> tensor<?x32xf32>
  return %written : tensor<?x32xf32>
}
func.func @compute(%a: memref<?x48xi8>, %w0: memref<1536xi8>, %w1: memref<1584xi8>, %out: memref<?x32xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<?x48xi8> to tensor<?x48xi8>
  %wt0 = bufferization.to_tensor %w0 restrict : memref<1536xi8> to tensor<1536xi8>
  %wt1 = bufferization.to_tensor %w1 restrict : memref<1584xi8> to tensor<1584xi8>
  %ot = bufferization.to_tensor %out restrict writable : memref<?x32xf32> to tensor<?x32xf32>
  %value = call @pipeline(%at, %wt0, %wt1, %ot) : (tensor<?x48xi8>, tensor<1536xi8>, tensor<1584xi8>, tensor<?x32xf32>) -> tensor<?x32xf32>
  bufferization.materialize_in_destination %value in restrict writable %out : (tensor<?x32xf32>, memref<?x32xf32>) -> ()
  return
}
