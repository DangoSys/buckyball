#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @compute(%a: memref<68x16xf32>, %w: memref<1x16x16xi8>, %mult: memref<68x1xf32>, %as: memref<68x1xf32>, %ws: memref<16x1xf32>, %out: memref<68x16xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<68x16xf32> to tensor<68x16xf32>
  %wl = bufferization.to_tensor %w restrict : memref<1x16x16xi8> to tensor<1x16x16xi8>
  %wg = tensor.expand_shape %wl [[0], [1], [2, 3]] output_shape [1, 16, 1, 16] : tensor<1x16x16xi8> into tensor<1x16x1x16xi8>
  %empty_w = tensor.empty() : tensor<1x16x1x16xi8>
  %wr = linalg.transpose ins(%wg : tensor<1x16x1x16xi8>) outs(%empty_w : tensor<1x16x1x16xi8>) permutation = [0, 3, 2, 1]
  %wt = tensor.collapse_shape %wr [[0, 1], [2, 3]] : tensor<1x16x1x16xi8> into tensor<16x16xi8>
  %mt = arith.constant dense<[[4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0], [1.0], [4.0], [2.0]]> : tensor<68x1xf32>
  %ast = arith.constant dense<[[0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5], [1.0], [0.25], [0.5]]> : tensor<68x1xf32>
  %wst = arith.constant dense<[[1.0], [0.5], [1.0], [0.5], [1.0], [0.5], [1.0], [0.5], [1.0], [0.5], [1.0], [0.5], [1.0], [0.5], [1.0], [0.5]]> : tensor<16x1xf32>
  %aq = "frontend.fp32_to_int8"(%at, %mt) : (tensor<68x16xf32>, tensor<68x1xf32>) -> tensor<68x16xi8>
  %ad = "frontend.int8_to_fp32"(%aq, %ast) : (tensor<68x16xi8>, tensor<68x1xf32>) -> tensor<68x16xf32>
  %wd = "frontend.int8_to_fp32"(%wt, %wst) : (tensor<16x16xi8>, tensor<16x1xf32>) -> tensor<16x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<68x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<68x16xf32>) -> tensor<68x16xf32>
  %product = linalg.matmul indexing_maps = [#a, #w, #c] ins(%ad, %wd : tensor<68x16xf32>, tensor<16x16xf32>) outs(%init : tensor<68x16xf32>) -> tensor<68x16xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<68x16xf32>, memref<68x16xf32>) -> ()
  return
}
