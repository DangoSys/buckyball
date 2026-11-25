func.func @compute(%a: memref<?x132xi8>, %w: memref<1x8x132xi8>, %out: memref<?x8xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<?x132xi8> to tensor<?x132xi8>
  %wt = bufferization.to_tensor %w restrict : memref<1x8x132xi8> to tensor<1x8x132xi8>
  %ad = "frontend.mxint8_to_fp32"(%at) {axis = 1 : i64} : (tensor<?x132xi8>) -> tensor<?x128xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 2 : i64} : (tensor<1x8x132xi8>) -> tensor<1x8x128xf32>
  %we = tensor.empty() : tensor<8x1x128xf32>
  %blocks = linalg.transpose ins(%wd : tensor<1x8x128xf32>) outs(%we : tensor<8x1x128xf32>) permutation = [1, 0, 2]
  %flat = tensor.collapse_shape %blocks [[0], [1, 2]] : tensor<8x1x128xf32> into tensor<8x128xf32>
  %te = tensor.empty() : tensor<128x8xf32>
  %b = linalg.transpose ins(%flat : tensor<8x128xf32>) outs(%te : tensor<128x8xf32>) permutation = [1, 0]
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %at, %c0 : tensor<?x132xi8>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty(%rows) : tensor<?x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x8xf32>) -> tensor<?x8xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<?x128xf32>, tensor<128x8xf32>) outs(%init : tensor<?x8xf32>) -> tensor<?x8xf32>
  bufferization.materialize_in_destination %product in restrict writable %out : (tensor<?x8xf32>, memref<?x8xf32>) -> ()
  return
}
