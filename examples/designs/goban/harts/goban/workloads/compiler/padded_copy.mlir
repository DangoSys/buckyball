func.func @compute(%a: memref<2x20xf32>, %w: memref<2x16x32xi8>, %mult: memref<2x1xf32>, %as: memref<2x1xf32>, %ws: memref<20x1xf32>, %out: memref<2x20xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x20xf32> to tensor<2x20xf32>
  %wl = bufferization.to_tensor %w restrict : memref<2x16x32xi8> to tensor<2x16x32xi8>
  %wg = tensor.expand_shape %wl [[0], [1], [2, 3]] output_shape [2, 16, 2, 16] : tensor<2x16x32xi8> into tensor<2x16x2x16xi8>
  %wre = tensor.empty() : tensor<2x16x2x16xi8>
  %wr = linalg.transpose ins(%wg : tensor<2x16x2x16xi8>) outs(%wre : tensor<2x16x2x16xi8>) permutation = [0, 3, 2, 1]
  %wp = tensor.collapse_shape %wr [[0, 1], [2, 3]] : tensor<2x16x2x16xi8> into tensor<32x32xi8>
  %wt = tensor.extract_slice %wp[0, 0] [20, 20] [1, 1] : tensor<32x32xi8> to tensor<20x20xi8>
  %mt = bufferization.to_tensor %mult restrict : memref<2x1xf32> to tensor<2x1xf32>
  %ast = bufferization.to_tensor %as restrict : memref<2x1xf32> to tensor<2x1xf32>
  %wst = bufferization.to_tensor %ws restrict : memref<20x1xf32> to tensor<20x1xf32>
  %aq = "frontend.fp32_to_int8"(%at, %mt) : (tensor<2x20xf32>, tensor<2x1xf32>) -> tensor<2x20xi8>
  %ad = "frontend.int8_to_fp32"(%aq, %ast) : (tensor<2x20xi8>, tensor<2x1xf32>) -> tensor<2x20xf32>
  %wd = "frontend.int8_to_fp32"(%wt, %wst) : (tensor<20x20xi8>, tensor<20x1xf32>) -> tensor<20x20xf32>
  %we = tensor.empty() : tensor<20x20xf32>
  %b = linalg.transpose ins(%wd : tensor<20x20xf32>) outs(%we : tensor<20x20xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x20xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x20xf32>) -> tensor<2x20xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<2x20xf32>, tensor<20x20xf32>) outs(%c : tensor<2x20xf32>) -> tensor<2x20xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x20xf32>, memref<2x20xf32>) -> ()
  return
}
