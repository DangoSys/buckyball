// CHECK-DAG: func.func private @{{.*}}hart.parallel_group = "matrix_with_batch_rows0", hart.parallel_part = 0 : i64
// CHECK-DAG: func.func private @{{.*}}hart.parallel_group = "matrix_with_batch_rows0", hart.parallel_part = 1 : i64
// CHECK-DAG: linalg.batch_matmul
// CHECK-NOT: hart.parallel_group = "dynamic_matrix_with_batch
#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#c = affine_map<(m,n,k)->(m,n)>
func.func @matrix_with_batch(%input: tensor<128x528xi8>, %weights: tensor<1x16x528xi8>, %batch_input: tensor<1x1x528xi8>, %batch_weights: tensor<1x528x4xi8>, %out: tensor<128x16xf32>) -> (tensor<128x16xf32>, tensor<1x1x4xf32>) {
  %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<128x528xi8>) -> tensor<128x512xf32>
  %w = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x16x528xi8>) -> tensor<1x16x512xf32>
  %we = tensor.empty() : tensor<16x1x512xf32>
  %wt = linalg.transpose ins(%w : tensor<1x16x512xf32>) outs(%we : tensor<16x1x512xf32>) permutation = [1,0,2]
  %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<16x1x512xf32> into tensor<16x512xf32>
  %c0 = arith.constant 0 : index
  %m = tensor.dim %input, %c0 : tensor<128x528xi8>
  %ce = tensor.empty() : tensor<128x16xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<128x16xf32>) -> tensor<128x16xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%a,%wc : tensor<128x512xf32>,tensor<16x512xf32>) outs(%init : tensor<128x16xf32>) -> tensor<128x16xf32>
  %ba = "frontend.mxint8_to_fp32"(%batch_input) {axis = 2 : i64} : (tensor<1x1x528xi8>) -> tensor<1x1x512xf32>
  %bw = "frontend.mxint8_to_fp32"(%batch_weights) {axis = 1 : i64} : (tensor<1x528x4xi8>) -> tensor<1x512x4xf32>
  %be = tensor.empty() : tensor<1x1x4xf32>
  %bi = linalg.fill ins(%zero : f32) outs(%be : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
  %batch = linalg.batch_matmul ins(%ba,%bw : tensor<1x1x512xf32>,tensor<1x512x4xf32>) outs(%bi : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
  %written = bufferization.materialize_in_destination %product in %out : (tensor<128x16xf32>,tensor<128x16xf32>) -> tensor<128x16xf32>
  return %written, %batch : tensor<128x16xf32>, tensor<1x1x4xf32>
}

func.func @dynamic_matrix_with_batch(%input: tensor<?x528xi8>, %weights: tensor<1x16x528xi8>, %batch_input: tensor<1x1x528xi8>, %batch_weights: tensor<1x528x4xi8>, %out: tensor<?x16xf32>) -> (tensor<?x16xf32>, tensor<1x1x4xf32>) {
  %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<?x528xi8>) -> tensor<?x512xf32>
  %w = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x16x528xi8>) -> tensor<1x16x512xf32>
  %we = tensor.empty() : tensor<16x1x512xf32>
  %wt = linalg.transpose ins(%w : tensor<1x16x512xf32>) outs(%we : tensor<16x1x512xf32>) permutation = [1,0,2]
  %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<16x1x512xf32> into tensor<16x512xf32>
  %c0 = arith.constant 0 : index
  %m = tensor.dim %input, %c0 : tensor<?x528xi8>
  %ce = tensor.empty(%m) : tensor<?x16xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<?x16xf32>) -> tensor<?x16xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%a,%wc : tensor<?x512xf32>,tensor<16x512xf32>) outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %ba = "frontend.mxint8_to_fp32"(%batch_input) {axis = 2 : i64} : (tensor<1x1x528xi8>) -> tensor<1x1x512xf32>
  %bw = "frontend.mxint8_to_fp32"(%batch_weights) {axis = 1 : i64} : (tensor<1x528x4xi8>) -> tensor<1x512x4xf32>
  %be = tensor.empty() : tensor<1x1x4xf32>
  %bi = linalg.fill ins(%zero : f32) outs(%be : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
  %batch = linalg.batch_matmul ins(%ba,%bw : tensor<1x1x512xf32>,tensor<1x512x4xf32>) outs(%bi : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
  %written = bufferization.materialize_in_destination %product in %out : (tensor<?x16xf32>,tensor<?x16xf32>) -> tensor<?x16xf32>
  return %written, %batch : tensor<?x16xf32>, tensor<1x1x4xf32>
}
