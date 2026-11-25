// CHECK-LABEL: func.func private @project(
// CHECK: hart.launch @project_ht0{{.*}}thread = 0
// CHECK: hart.launch @project_ht1{{.*}}thread = 1
// CHECK: hart.launch @project_ht2{{.*}}thread = 2
// CHECK: hart.launch @project_ht3{{.*}}thread = 3
// CHECK: hart.launch @project_ht4{{.*}}thread = 4
// CHECK-NOT: hart.launch
// CHECK: return
// BATCH-LABEL: func.func @compute_columns(
// BATCH-DAG: %[[JOBS:.*]] = arith.constant 5 : i64
// BATCH-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// BATCH: call @stream_begin(%[[JOBS]], %[[ZERO]], %[[ZERO]])
// BATCH-NOT: call @stream_wait
// BATCH-NOT: memref.copy
// BATCH: call @project_ht4_launch
// BATCH: call @stream_submit()
// BATCH: call @stream_wait()
// BATCH-NOT: call @stream_begin
// BATCH-NOT: memref.copy
// BATCH-NOT: linalg.copy
// BATCH-NOT: call @stream_copy
// BATCH: return
#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#o = affine_map<(m,n,k)->(m,n)>
func.func private @project(%a: tensor<?x528xi8> {bufferization.buffer_layout = strided<[528,1], offset: ?>}, %w: tensor<1x160x528xi8> {bufferization.buffer_layout = strided<[84480,528,1], offset: ?>}) -> tensor<?x160xf32> {
  %x = "frontend.mxint8_to_fp32"(%a) {axis = 1 : i64} : (tensor<?x528xi8>) -> tensor<?x512xf32>
  %y = "frontend.mxint8_to_fp32"(%w) {axis = 2 : i64} : (tensor<1x160x528xi8>) -> tensor<1x160x512xf32>
  %t = tensor.empty() : tensor<160x1x512xf32>
  %transposed = linalg.transpose ins(%y : tensor<1x160x512xf32>) outs(%t : tensor<160x1x512xf32>) permutation = [1,0,2]
  %weights = tensor.collapse_shape %transposed [[0],[1,2]] : tensor<160x1x512xf32> into tensor<160x512xf32>
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %a, %c0 : tensor<?x528xi8>
  %empty = tensor.empty(%rows) : tensor<?x160xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x160xf32>) -> tensor<?x160xf32>
  %value = linalg.matmul indexing_maps=[#a,#w,#o] ins(%x,%weights : tensor<?x512xf32>,tensor<160x512xf32>) outs(%init : tensor<?x160xf32>) -> tensor<?x160xf32>
  return %value : tensor<?x160xf32>
}
func.func @compute_columns(%a: memref<?x528xi8, strided<[528,1], offset: ?>>, %w: memref<1x160x528xi8, strided<[84480,528,1], offset: ?>>, %out: memref<?x160xf32, strided<[160,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %a restrict : memref<?x528xi8, strided<[528,1], offset: ?>> to tensor<?x528xi8>
  %y = bufferization.to_tensor %w restrict : memref<1x160x528xi8, strided<[84480,528,1], offset: ?>> to tensor<1x160x528xi8>
  %value = func.call @project(%x,%y) : (tensor<?x528xi8>,tensor<1x160x528xi8>) -> tensor<?x160xf32>
  bufferization.materialize_in_destination %value in restrict writable %out : (tensor<?x160xf32>,memref<?x160xf32, strided<[160,1], offset: ?>>) -> ()
  return
}
