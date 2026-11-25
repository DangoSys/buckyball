// Three K blocks compete for four banks: retain one A block, stream the rest.
// PARTIAL: %[[A:.*]] = bbacc.bank_mvin {{.*}} : memref<1x1x2x1056xi8,
// PARTIAL: scf.for
// PARTIAL: mxint8matmul_ball.bank_matmul_mxint8 %[[A]],
// PARTIAL-NOT: bbacc.bank_release %[[A]]
// PARTIAL: bbacc.bank_mvout
// PARTIAL: bbacc.bank_release %[[A]]
func.func @compute(%a: memref<3x3072xf32>, %w: memref<3x68x1056xi8>, %out: memref<3x67xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<3x3072xf32> to tensor<3x3072xf32>
  %wp = bufferization.to_tensor %w restrict : memref<3x68x1056xi8> to tensor<3x68x1056xi8>
  %wt = tensor.extract_slice %wp[0, 0, 0] [3, 67, 1056] [1, 1, 1] : tensor<3x68x1056xi8> to tensor<3x67x1056xi8>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<3x3072xf32>) -> tensor<3x3168xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<3x3168xi8>) -> tensor<3x3072xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 2 : i64} : (tensor<3x67x1056xi8>) -> tensor<3x67x1024xf32>
  %rows = tensor.empty() : tensor<67x3x1024xf32>
  %ordered = linalg.transpose ins(%wd : tensor<3x67x1024xf32>) outs(%rows : tensor<67x3x1024xf32>) permutation = [1, 0, 2]
  %flat = tensor.collapse_shape %ordered [[0], [1, 2]] : tensor<67x3x1024xf32> into tensor<67x3072xf32>
  %we = tensor.empty() : tensor<3072x67xf32>
  %b = linalg.transpose ins(%flat : tensor<67x3072xf32>) outs(%we : tensor<3072x67xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<3x67xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<3x67xf32>) -> tensor<3x67xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<3x3072xf32>, tensor<3072x67xf32>) outs(%c : tensor<3x67xf32>) -> tensor<3x67xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<3x67xf32>, memref<3x67xf32>) -> ()
  return
}
