func.func @compute(%a: memref<2x3x64xf32>, %w: memref<2x64x?xf32>, %out: memref<2x3x?xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x3x64xf32> to tensor<2x3x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x64x?xf32> to tensor<2x64x?xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x3x64xf32>) -> tensor<2x3x66xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x3x66xi8>) -> tensor<2x3x64xf32>
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2 : index
  %n = tensor.dim %wt, %two : tensor<2x64x?xf32>
  %ce = tensor.empty(%n) : tensor<2x3x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  %product = linalg.batch_matmul ins(%ad, %wt : tensor<2x3x64xf32>, tensor<2x64x?xf32>) outs(%c : tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x3x?xf32>, memref<2x3x?xf32>) -> ()
  return
}
