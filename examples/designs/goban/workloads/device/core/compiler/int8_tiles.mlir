func.func @compute(%a: memref<3x273xf32>, %w: memref<2x32x144xi8>, %mult: memref<3x1xf32>, %as: memref<3x1xf32>, %ws: memref<19x1xf32>, %out: memref<3x19xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<3x273xf32> to tensor<3x273xf32>
  // The weight in the smatmul B layout: 2 slices of 16 channels, K padded to
  // 288 in 9 groups of 32 lines.
  %wl = bufferization.to_tensor %w restrict : memref<2x32x144xi8> to tensor<2x32x144xi8>
  %wg = tensor.expand_shape %wl [[0], [1], [2, 3]] output_shape [2, 32, 9, 16] : tensor<2x32x144xi8> into tensor<2x32x9x16xi8>
  %wre = tensor.empty() : tensor<2x16x9x32xi8>
  %wr = linalg.transpose ins(%wg : tensor<2x32x9x16xi8>) outs(%wre : tensor<2x16x9x32xi8>) permutation = [0, 3, 2, 1]
  %wp = tensor.collapse_shape %wr [[0, 1], [2, 3]] : tensor<2x16x9x32xi8> into tensor<32x288xi8>
  %wt = tensor.extract_slice %wp[0, 0] [19, 273] [1, 1] : tensor<32x288xi8> to tensor<19x273xi8>
  %mt = bufferization.to_tensor %mult restrict : memref<3x1xf32> to tensor<3x1xf32>
  %ast = bufferization.to_tensor %as restrict : memref<3x1xf32> to tensor<3x1xf32>
  %wst = bufferization.to_tensor %ws restrict : memref<19x1xf32> to tensor<19x1xf32>
  %aq = "frontend.fp32_to_int8"(%at, %mt) : (tensor<3x273xf32>, tensor<3x1xf32>) -> tensor<3x273xi8>
  %ad = "frontend.int8_to_fp32"(%aq, %ast) : (tensor<3x273xi8>, tensor<3x1xf32>) -> tensor<3x273xf32>
  %wd = "frontend.int8_to_fp32"(%wt, %wst) : (tensor<19x273xi8>, tensor<19x1xf32>) -> tensor<19x273xf32>
  %we = tensor.empty() : tensor<273x19xf32>
  %b = linalg.transpose ins(%wd : tensor<19x273xf32>) outs(%we : tensor<273x19xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<3x19xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<3x19xf32>) -> tensor<3x19xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<3x273xf32>, tensor<273x19xf32>) outs(%c : tensor<3x19xf32>) -> tensor<3x19xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<3x19xf32>, memref<3x19xf32>) -> ()
  return
}
