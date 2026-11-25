func.func @compute(%a: memref<37x768xf32>, %w: memref<6x120x144xi8>, %out: memref<37x120xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<37x768xf32> to tensor<37x768xf32>
  %wt = bufferization.to_tensor %w restrict : memref<6x120x144xi8> to tensor<6x120x144xi8>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<37x768xf32>) -> tensor<37x792xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<37x792xi8>) -> tensor<37x768xf32>
  %blocks = tensor.extract_slice %wt[0, 0, 0] [6, 120, 132] [1, 1, 1] : tensor<6x120x144xi8> to tensor<6x120x132xi8>
  %decoded = "frontend.mxint8_to_fp32"(%blocks) {axis = 2 : i64} : (tensor<6x120x132xi8>) -> tensor<6x120x128xf32>
  %we = tensor.empty() : tensor<6x128x120xf32>
  %transposed = linalg.transpose ins(%decoded : tensor<6x120x128xf32>) outs(%we : tensor<6x128x120xf32>) permutation = [0, 2, 1]
  %b = tensor.collapse_shape %transposed [[0, 1], [2]] : tensor<6x128x120xf32> into tensor<768x120xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<37x120xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<37x120xf32>) -> tensor<37x120xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<37x768xf32>, tensor<768x120xf32>) outs(%c : tensor<37x120xf32>) -> tensor<37x120xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<37x120xf32>, memref<37x120xf32>) -> ()
  return
}
