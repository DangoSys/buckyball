// The softmax of the frontend's decomposition over the rows of [8, 64]:
// convert-linalg-to-hart replaces it by one SoftmaxBall block.
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
func.func @compute(%x: memref<8x64xf32>, %out: memref<8x64xf32>) attributes {llvm.emit_c_interface} {
  %xt = bufferization.to_tensor %x restrict : memref<8x64xf32> to tensor<8x64xf32>
  %lowest = arith.constant -3.40282347E+38 : f32
  %zero = arith.constant 0.0 : f32
  %re = tensor.empty() : tensor<8xf32>
  %mi = linalg.fill ins(%lowest : f32) outs(%re : tensor<8xf32>) -> tensor<8xf32>
  %m = linalg.reduce ins(%xt : tensor<8x64xf32>) outs(%mi : tensor<8xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.maximumf %in, %init : f32
      linalg.yield %v : f32
    }
  %me = tensor.expand_shape %m [[0, 1]] output_shape [8, 1] : tensor<8xf32> into tensor<8x1xf32>
  %e0 = tensor.empty() : tensor<8x64xf32>
  %d = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%xt, %me : tensor<8x64xf32>, tensor<8x1xf32>) outs(%e0 : tensor<8x64xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.subf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<8x64xf32>
  %e = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%d : tensor<8x64xf32>) outs(%e0 : tensor<8x64xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = math.exp %a : f32
      linalg.yield %v : f32
  } -> tensor<8x64xf32>
  %si = linalg.fill ins(%zero : f32) outs(%re : tensor<8xf32>) -> tensor<8xf32>
  %s = linalg.reduce ins(%e : tensor<8x64xf32>) outs(%si : tensor<8xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.addf %in, %init : f32
      linalg.yield %v : f32
    }
  %se = tensor.expand_shape %s [[0, 1]] output_shape [8, 1] : tensor<8xf32> into tensor<8x1xf32>
  %y = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%e, %se : tensor<8x64xf32>, tensor<8x1xf32>) outs(%e0 : tensor<8x64xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.divf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<8x64xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<8x64xf32>, memref<8x64xf32>) -> ()
  return
}
