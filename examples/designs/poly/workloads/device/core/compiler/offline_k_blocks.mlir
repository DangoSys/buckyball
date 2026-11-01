func.func @compute(%a: memref<3x2048xf32>, %w: memref<2x20x1056xi8>, %out: memref<3x19xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<3x2048xf32> to tensor<3x2048xf32>
  %wp = bufferization.to_tensor %w restrict : memref<2x20x1056xi8> to tensor<2x20x1056xi8>
  %wt = tensor.extract_slice %wp[0, 0, 0] [2, 19, 1056] [1, 1, 1] : tensor<2x20x1056xi8> to tensor<2x19x1056xi8>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<3x2048xf32>) -> tensor<3x2112xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<3x2112xi8>) -> tensor<3x2048xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 2 : i64} : (tensor<2x19x1056xi8>) -> tensor<2x19x1024xf32>
  %rows = tensor.empty() : tensor<19x2x1024xf32>
  %ordered = linalg.transpose ins(%wd : tensor<2x19x1024xf32>) outs(%rows : tensor<19x2x1024xf32>) permutation = [1, 0, 2]
  %flat = tensor.collapse_shape %ordered [[0], [1, 2]] : tensor<19x2x1024xf32> into tensor<19x2048xf32>
  %we = tensor.empty() : tensor<2048x19xf32>
  %b = linalg.transpose ins(%flat : tensor<19x2048xf32>) outs(%we : tensor<2048x19xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<3x19xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<3x19xf32>) -> tensor<3x19xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<3x2048xf32>, tensor<2048x19xf32>) outs(%c : tensor<3x19xf32>) -> tensor<3x19xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<3x19xf32>, memref<3x19xf32>) -> ()
  return
}
