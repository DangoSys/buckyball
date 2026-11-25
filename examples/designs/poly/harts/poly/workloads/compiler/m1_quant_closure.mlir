// CHECK-LABEL: func.func private @m1_ht0(
// CHECK: %[[PRODUCT:.*]] = linalg.matmul
// CHECK-NEXT: %[[PACKED:.*]] = "frontend.fp32_to_mxint8"(%[[PRODUCT]])
// CHECK: bufferization.materialize_in_destination %[[PACKED]]
// CHECK: return
// CHECK-LABEL: func.func private @prefill_ht0(
// CHECK: linalg.matmul
// CHECK-NOT: frontend.fp32_to_mxint8
// CHECK: return
// CHECK-LABEL: func.func private @dynamic_ht0(
// CHECK: linalg.matmul
// CHECK-NOT: frontend.fp32_to_mxint8
// CHECK: return
// CHECK-LABEL: func.func private @fanout_ht0(
// CHECK: linalg.matmul
// CHECK-NOT: frontend.fp32_to_mxint8
// CHECK: return
#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#c = affine_map<(m,n,k)->(m,n)>
func.func @m1(%input: tensor<1x1056xi8>, %weights: tensor<1x32x1056xi8>, %out: tensor<1x33xi8>) -> tensor<1x33xi8> {
  %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<1x1056xi8>) -> tensor<1x1024xf32>
  %w = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x32x1056xi8>) -> tensor<1x32x1024xf32>
  %we = tensor.empty() : tensor<32x1x1024xf32>
  %wt = linalg.transpose ins(%w : tensor<1x32x1024xf32>) outs(%we : tensor<32x1x1024xf32>) permutation = [1,0,2]
  %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<32x1x1024xf32> into tensor<32x1024xf32>
  %ce = tensor.empty() : tensor<1x32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<1x32xf32>) -> tensor<1x32xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%a,%wc : tensor<1x1024xf32>,tensor<32x1024xf32>) outs(%init : tensor<1x32xf32>) -> tensor<1x32xf32>
  %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<1x32xf32>) -> tensor<1x33xi8>
  %written = bufferization.materialize_in_destination %packed in %out : (tensor<1x33xi8>,tensor<1x33xi8>) -> tensor<1x33xi8>
  return %written : tensor<1x33xi8>
}
func.func @prefill(%input: tensor<128x1056xi8>, %weights: tensor<1x32x1056xi8>, %out: tensor<128x33xi8>) -> tensor<128x33xi8> {
  %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<128x1056xi8>) -> tensor<128x1024xf32>
  %w = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x32x1056xi8>) -> tensor<1x32x1024xf32>
  %we = tensor.empty() : tensor<32x1x1024xf32>
  %wt = linalg.transpose ins(%w : tensor<1x32x1024xf32>) outs(%we : tensor<32x1x1024xf32>) permutation = [1,0,2]
  %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<32x1x1024xf32> into tensor<32x1024xf32>
  %ce = tensor.empty() : tensor<128x32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<128x32xf32>) -> tensor<128x32xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%a,%wc : tensor<128x1024xf32>,tensor<32x1024xf32>) outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
  %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<128x32xf32>) -> tensor<128x33xi8>
  %written = bufferization.materialize_in_destination %packed in %out : (tensor<128x33xi8>,tensor<128x33xi8>) -> tensor<128x33xi8>
  return %written : tensor<128x33xi8>
}
func.func @dynamic(%input: tensor<?x1056xi8>, %weights: tensor<1x32x1056xi8>, %out: tensor<?x33xi8>) -> tensor<?x33xi8> {
  %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<?x1056xi8>) -> tensor<?x1024xf32>
  %w = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x32x1056xi8>) -> tensor<1x32x1024xf32>
  %we = tensor.empty() : tensor<32x1x1024xf32>
  %wt = linalg.transpose ins(%w : tensor<1x32x1024xf32>) outs(%we : tensor<32x1x1024xf32>) permutation = [1,0,2]
  %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<32x1x1024xf32> into tensor<32x1024xf32>
  %c0 = arith.constant 0 : index
  %m = tensor.dim %input, %c0 : tensor<?x1056xi8>
  %ce = tensor.empty(%m) : tensor<?x32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<?x32xf32>) -> tensor<?x32xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%a,%wc : tensor<?x1024xf32>,tensor<32x1024xf32>) outs(%init : tensor<?x32xf32>) -> tensor<?x32xf32>
  %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<?x32xf32>) -> tensor<?x33xi8>
  %written = bufferization.materialize_in_destination %packed in %out : (tensor<?x33xi8>,tensor<?x33xi8>) -> tensor<?x33xi8>
  return %written : tensor<?x33xi8>
}
func.func @fanout(%input: tensor<1x1056xi8>, %weights: tensor<1x32x1056xi8>, %out: tensor<1x33xi8>) -> (tensor<1x32xf32>, tensor<1x33xi8>) {
  %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<1x1056xi8>) -> tensor<1x1024xf32>
  %w = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x32x1056xi8>) -> tensor<1x32x1024xf32>
  %we = tensor.empty() : tensor<32x1x1024xf32>
  %wt = linalg.transpose ins(%w : tensor<1x32x1024xf32>) outs(%we : tensor<32x1x1024xf32>) permutation = [1,0,2]
  %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<32x1x1024xf32> into tensor<32x1024xf32>
  %ce = tensor.empty() : tensor<1x32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<1x32xf32>) -> tensor<1x32xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%a,%wc : tensor<1x1024xf32>,tensor<32x1024xf32>) outs(%init : tensor<1x32xf32>) -> tensor<1x32xf32>
  %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<1x32xf32>) -> tensor<1x33xi8>
  %written = bufferization.materialize_in_destination %packed in %out : (tensor<1x33xi8>,tensor<1x33xi8>) -> tensor<1x33xi8>
  return %product, %written : tensor<1x32xf32>, tensor<1x33xi8>
}
