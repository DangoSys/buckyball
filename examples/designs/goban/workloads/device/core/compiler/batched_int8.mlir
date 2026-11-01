func.func @compute(%a: memref<2x3x32xf32>, %w: memref<2x32x16xf32>, %out: memref<2x3x16xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x3x32xf32> to tensor<2x3x32xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x32x16xf32> to tensor<2x32x16xf32>
  %scale = arith.constant dense<1.0> : tensor<1x1x1xf32>
  %aq = "frontend.fp32_to_int8"(%at, %scale) : (tensor<2x3x32xf32>, tensor<1x1x1xf32>) -> tensor<2x3x32xi8>
  %wq = "frontend.fp32_to_int8"(%wt, %scale) : (tensor<2x32x16xf32>, tensor<1x1x1xf32>) -> tensor<2x32x16xi8>
  %ad = "frontend.int8_to_fp32"(%aq, %scale) : (tensor<2x3x32xi8>, tensor<1x1x1xf32>) -> tensor<2x3x32xf32>
  %wd = "frontend.int8_to_fp32"(%wq, %scale) : (tensor<2x32x16xi8>, tensor<1x1x1xf32>) -> tensor<2x32x16xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x3x16xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x3x16xf32>) -> tensor<2x3x16xf32>
  %product = linalg.batch_matmul ins(%ad, %wd : tensor<2x3x32xf32>, tensor<2x32x16xf32>) outs(%c : tensor<2x3x16xf32>) -> tensor<2x3x16xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x3x16xf32>, memref<2x3x16xf32>) -> ()
  return
}
