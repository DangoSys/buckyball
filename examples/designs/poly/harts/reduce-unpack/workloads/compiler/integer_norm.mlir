// CHECK: intnorm_ball.bank_norm_mxint8
#map = affine_map<(i,j)->(i,j)>
func.func private @add(%a: tensor<?x1024xf32>, %b: tensor<?x1024xf32>) -> tensor<?x1024xf32> {
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %a, %c0 : tensor<?x1024xf32>
  %empty = tensor.empty(%rows) : tensor<?x1024xf32>
  %sum = linalg.generic {indexing_maps=[#map,#map,#map],iterator_types=["parallel","parallel"]}
      ins(%a,%b : tensor<?x1024xf32>,tensor<?x1024xf32>) outs(%empty : tensor<?x1024xf32>) {
  ^bb0(%x:f32,%y:f32,%unused:f32):
    %z = arith.addf %x,%y : f32
    linalg.yield %z : f32
  } -> tensor<?x1024xf32>
  return %sum : tensor<?x1024xf32>
}
func.func @compute(%input: memref<?x1024xf32, strided<[1024, 1], offset: ?>>,
    %coefficients: memref<8208xi8, strided<[1], offset: ?>>,
    %output: memref<?x1056xi8, strided<[1056, 1], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<?x1024xf32, strided<[1024, 1], offset: ?>> to tensor<?x1024xf32>
  %p = bufferization.to_tensor %coefficients restrict : memref<8208xi8, strided<[1], offset: ?>> to tensor<8208xi8>
  %s1 = func.call @add(%x,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %s2 = func.call @add(%s1,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %s3 = func.call @add(%s2,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %s4 = func.call @add(%s3,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %s5 = func.call @add(%s4,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %s6 = func.call @add(%s5,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %s7 = func.call @add(%s6,%x) : (tensor<?x1024xf32>,tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %s7, %c0 : tensor<?x1024xf32>
  %expanded = tensor.expand_shape %s7 [[0], [1, 2]] output_shape [%rows, 1, 1024] : tensor<?x1024xf32> into tensor<?x1x1024xf32>
  %target = tensor.empty(%rows) : tensor<1x?x1024xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<?x1x1024xf32>) outs(%target : tensor<1x?x1024xf32>) permutation = [1,0,2]
  %view = tensor.collapse_shape %transposed [[0,1], [2]] : tensor<1x?x1024xf32> into tensor<?x1024xf32>
  %y = "frontend.fixed_norm_mxint8"(%view, %p) : (tensor<?x1024xf32>, tensor<8208xi8>) -> tensor<?x1056xi8>
  bufferization.materialize_in_destination %y in writable %output : (tensor<?x1056xi8>, memref<?x1056xi8, strided<[1056, 1], offset: ?>>) -> ()
  return
}
