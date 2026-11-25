func.func @compute(%a: memref<33x32xf32>, %w: memref<256x48xi8>, %out: memref<33x256xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<33x32xf32> to tensor<33x32xf32>
  %wt = bufferization.to_tensor %w restrict : memref<256x48xi8> to tensor<256x48xi8>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<33x32xf32>) -> tensor<33x33xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<33x33xi8>) -> tensor<33x32xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 1 : i64} : (tensor<256x48xi8>) -> tensor<256x32xf32>
  %empty = tensor.empty() : tensor<32x256xf32>
  %b = linalg.transpose ins(%wd : tensor<256x32xf32>) outs(%empty : tensor<32x256xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<33x256xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<33x256xf32>) -> tensor<33x256xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<33x32xf32>, tensor<32x256xf32>) outs(%c : tensor<33x256xf32>) -> tensor<33x256xf32>
  bufferization.materialize_in_destination %product in restrict writable %out : (tensor<33x256xf32>, memref<33x256xf32>) -> ()
  return
}
