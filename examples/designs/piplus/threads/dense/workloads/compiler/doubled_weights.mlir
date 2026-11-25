// A linear layer of two MXINT8 weight terms, as the recipe stores every
// Linear: C = A . W1^T + bias + A . W2^T, both weights in the mxint8mm layout
// (1 block(s) of 768). convert-linalg-to-hart starts the first product at the
// bias and the second at the first, and the thread accumulates both terms
// onto each C tile; M = 37 is a tile and a part.
func.func @compute(%a: memref<37x768xf32>, %w1: memref<1x120x800xi8>, %w2: memref<1x120x800xi8>, %bias: memref<1x120xf32>, %out: memref<37x120xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<37x768xf32> to tensor<37x768xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<37x768xf32>) -> tensor<37x792xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<37x792xi8>) -> tensor<37x768xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<37x120xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<37x120xf32>) -> tensor<37x120xf32>
  %we = tensor.empty() : tensor<1x768x120xf32>
  %w1t = bufferization.to_tensor %w1 restrict : memref<1x120x800xi8> to tensor<1x120x800xi8>
  %w1s = tensor.extract_slice %w1t[0, 0, 0] [1, 120, 792] [1, 1, 1] : tensor<1x120x800xi8> to tensor<1x120x792xi8>
  %w1d = "frontend.mxint8_to_fp32"(%w1s) {axis = 2 : i64} : (tensor<1x120x792xi8>) -> tensor<1x120x768xf32>
  %w1p = linalg.transpose ins(%w1d : tensor<1x120x768xf32>) outs(%we : tensor<1x768x120xf32>) permutation = [0, 2, 1]
  %b1 = tensor.collapse_shape %w1p [[0, 1], [2]] : tensor<1x768x120xf32> into tensor<768x120xf32>
  %p1 = linalg.matmul ins(%ad, %b1 : tensor<37x768xf32>, tensor<768x120xf32>) outs(%c : tensor<37x120xf32>) -> tensor<37x120xf32>
  %bt = bufferization.to_tensor %bias restrict : memref<1x120xf32> to tensor<1x120xf32>
  %biased = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%bt, %p1 : tensor<1x120xf32>, tensor<37x120xf32>) outs(%ce : tensor<37x120xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
  } -> tensor<37x120xf32>
  %w2t = bufferization.to_tensor %w2 restrict : memref<1x120x800xi8> to tensor<1x120x800xi8>
  %w2s = tensor.extract_slice %w2t[0, 0, 0] [1, 120, 792] [1, 1, 1] : tensor<1x120x800xi8> to tensor<1x120x792xi8>
  %w2d = "frontend.mxint8_to_fp32"(%w2s) {axis = 2 : i64} : (tensor<1x120x792xi8>) -> tensor<1x120x768xf32>
  %w2p = linalg.transpose ins(%w2d : tensor<1x120x768xf32>) outs(%we : tensor<1x768x120xf32>) permutation = [0, 2, 1]
  %b2 = tensor.collapse_shape %w2p [[0, 1], [2]] : tensor<1x768x120xf32> into tensor<768x120xf32>
  %p2 = linalg.matmul ins(%ad, %b2 : tensor<37x768xf32>, tensor<768x120xf32>) outs(%c : tensor<37x120xf32>) -> tensor<37x120xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%biased, %p2 : tensor<37x120xf32>, tensor<37x120xf32>) outs(%ce : tensor<37x120xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
  } -> tensor<37x120xf32>
  bufferization.materialize_in_destination %sum in writable %out : (tensor<37x120xf32>, memref<37x120xf32>) -> ()
  return
}
