func.func @compute(%a: memref<3x768xf32>, %w: memref<32x768xf32>, %out: memref<3x32xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<3x768xf32> to tensor<3x768xf32>
  %wt = bufferization.to_tensor %w restrict : memref<32x768xf32> to tensor<32x768xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<3x768xf32>) -> tensor<3x792xi8>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 1 : i64} : (tensor<32x768xf32>) -> tensor<32x792xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<3x792xi8>) -> tensor<3x768xf32>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 1 : i64} : (tensor<32x792xi8>) -> tensor<32x768xf32>
  %we = tensor.empty() : tensor<768x32xf32>
  %b = linalg.transpose ins(%wd : tensor<32x768xf32>) outs(%we : tensor<768x32xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<3x32xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<3x32xf32>) -> tensor<3x32xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<3x768xf32>, tensor<768x32xf32>) outs(%c : tensor<3x32xf32>) -> tensor<3x32xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<3x32xf32>, memref<3x32xf32>) -> ()
  return
}
