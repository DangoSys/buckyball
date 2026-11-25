#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @compute(%a: memref<12x3072xf32>, %w: memref<1x384x128xi8>, %out: memref<12x16xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<12x3072xf32> to tensor<12x3072xf32>
  %wl = bufferization.to_tensor %w restrict : memref<1x384x128xi8> to tensor<1x384x128xi8>
  %wg = tensor.expand_shape %wl [[0], [1], [2, 3]] output_shape [1, 384, 8, 16] : tensor<1x384x128xi8> into tensor<1x384x8x16xi8>
  %we = tensor.empty() : tensor<1x16x8x384xi8>
  %wr = linalg.transpose ins(%wg : tensor<1x384x8x16xi8>) outs(%we : tensor<1x16x8x384xi8>) permutation = [0, 3, 2, 1]
  %wt = tensor.collapse_shape %wr [[0, 1], [2, 3]] : tensor<1x16x8x384xi8> into tensor<16x3072xi8>
  %scale = arith.constant dense<1.0> : tensor<1x1xf32>
  %aq = "frontend.fp32_to_int8"(%at, %scale) : (tensor<12x3072xf32>, tensor<1x1xf32>) -> tensor<12x3072xi8>
  %ad = "frontend.int8_to_fp32"(%aq, %scale) : (tensor<12x3072xi8>, tensor<1x1xf32>) -> tensor<12x3072xf32>
  %wd = "frontend.int8_to_fp32"(%wt, %scale) : (tensor<16x3072xi8>, tensor<1x1xf32>) -> tensor<16x3072xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<12x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<12x16xf32>) -> tensor<12x16xf32>
  %product = linalg.matmul indexing_maps = [#a, #w, #c] ins(%ad, %wd : tensor<12x3072xf32>, tensor<16x3072xf32>) outs(%init : tensor<12x16xf32>) -> tensor<12x16xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<12x16xf32>, memref<12x16xf32>) -> ()
  return
}
