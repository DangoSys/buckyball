func.func @compute_head_view(%a: memref<2x?x?xf32>, %w: memref<?x1x64xf32>, %out: memref<2x?x32xf32>) attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c31 = arith.constant 31 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.0 : f32
  %at = bufferization.to_tensor %a restrict : memref<2x?x?xf32> to tensor<2x?x?xf32>
  %wt = bufferization.to_tensor %w restrict : memref<?x1x64xf32> to tensor<?x1x64xf32>
  %m = tensor.dim %at, %c1 : tensor<2x?x?xf32>
  %k = tensor.dim %wt, %c0 : tensor<?x1x64xf32>
  %sum = arith.addi %k, %c31 : index
  %blocks = arith.divui %sum, %c32 : index
  %padded = arith.muli %blocks, %c32 : index
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x?xf32>) -> tensor<2x?x?xi8>
  %af = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x?xi8>) -> tensor<2x?x?xf32>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 0 : i64} : (tensor<?x1x64xf32>) -> tensor<?x1x64xi8>
  %wf = "frontend.mxint8_to_fp32"(%wq) {axis = 0 : i64} : (tensor<?x1x64xi8>) -> tensor<?x1x64xf32>
  %flat = tensor.collapse_shape %wf [[0, 1, 2]] : tensor<?x1x64xf32> into tensor<?xf32>
  %heads = tensor.expand_shape %flat [[0, 1, 2]] output_shape [%padded, 2, 32] : tensor<?xf32> into tensor<?x2x32xf32>
  %we = tensor.empty(%padded) : tensor<2x?x32xf32>
  %transposed = linalg.transpose ins(%heads : tensor<?x2x32xf32>) outs(%we : tensor<2x?x32xf32>) permutation = [1, 0, 2]
  %empty = tensor.empty(%m) : tensor<2x?x32xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  %product = linalg.batch_matmul ins(%af, %transposed : tensor<2x?x?xf32>, tensor<2x?x32xf32>) outs(%init : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x32xf32>, memref<2x?x32xf32>) -> ()
  return
}
