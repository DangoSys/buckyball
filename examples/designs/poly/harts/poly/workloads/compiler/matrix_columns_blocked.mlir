#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#o = affine_map<(m,n,k)->(m,n)>
func.func private @project(%a: tensor<?x4224xi8> {bufferization.buffer_layout = strided<[4224,1], offset: ?>}, %w: tensor<2x160x2112xi8> {bufferization.buffer_layout = strided<[337920,2112,1], offset: ?>}) -> tensor<?x160xf32> {
  %x = "frontend.mxint8_to_fp32"(%a) {axis = 1 : i64} : (tensor<?x4224xi8>) -> tensor<?x4096xf32>
  %y = "frontend.mxint8_to_fp32"(%w) {axis = 2 : i64} : (tensor<2x160x2112xi8>) -> tensor<2x160x2048xf32>
  %t = tensor.empty() : tensor<160x2x2048xf32>
  %transposed = linalg.transpose ins(%y : tensor<2x160x2048xf32>) outs(%t : tensor<160x2x2048xf32>) permutation = [1,0,2]
  %weights = tensor.collapse_shape %transposed [[0],[1,2]] : tensor<160x2x2048xf32> into tensor<160x4096xf32>
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %a, %c0 : tensor<?x4224xi8>
  %empty = tensor.empty(%rows) : tensor<?x160xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x160xf32>) -> tensor<?x160xf32>
  %value = linalg.matmul indexing_maps=[#a,#w,#o] ins(%x,%weights : tensor<?x4096xf32>,tensor<160x4096xf32>) outs(%init : tensor<?x160xf32>) -> tensor<?x160xf32>
  return %value : tensor<?x160xf32>
}
func.func @compute_columns(%a: memref<?x4224xi8, strided<[4224,1], offset: ?>>, %w: memref<2x160x2112xi8, strided<[337920,2112,1], offset: ?>>, %out: memref<?x160xf32, strided<[160,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %a restrict : memref<?x4224xi8, strided<[4224,1], offset: ?>> to tensor<?x4224xi8>
  %y = bufferization.to_tensor %w restrict : memref<2x160x2112xi8, strided<[337920,2112,1], offset: ?>> to tensor<2x160x2112xi8>
  %value = func.call @project(%x,%y) : (tensor<?x4224xi8>,tensor<2x160x2112xi8>) -> tensor<?x160xf32>
  bufferization.materialize_in_destination %value in restrict writable %out : (tensor<?x160xf32>,memref<?x160xf32, strided<[160,1], offset: ?>>) -> ()
  return
}
