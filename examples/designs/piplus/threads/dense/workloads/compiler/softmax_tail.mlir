// The softmax of the frontend's decomposition over the rows of [20, 241]:
// convert-linalg-to-hart replaces it by SoftmaxBall blocks of ten rows,
// each row moved as its whole bank rows and its tail of one value.
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
func.func @compute(%x: memref<20x241xf32>, %out: memref<20x241xf32>) attributes {llvm.emit_c_interface} {
  %xt = bufferization.to_tensor %x restrict : memref<20x241xf32> to tensor<20x241xf32>
  %lowest = arith.constant -3.40282347E+38 : f32
  %zero = arith.constant 0.0 : f32
  %re = tensor.empty() : tensor<20xf32>
  %mi = linalg.fill ins(%lowest : f32) outs(%re : tensor<20xf32>) -> tensor<20xf32>
  %m = linalg.reduce ins(%xt : tensor<20x241xf32>) outs(%mi : tensor<20xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.maximumf %in, %init : f32
      linalg.yield %v : f32
    }
  %me = tensor.expand_shape %m [[0, 1]] output_shape [20, 1] : tensor<20xf32> into tensor<20x1xf32>
  %e0 = tensor.empty() : tensor<20x241xf32>
  %d = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%xt, %me : tensor<20x241xf32>, tensor<20x1xf32>) outs(%e0 : tensor<20x241xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.subf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x241xf32>
  %e = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%d : tensor<20x241xf32>) outs(%e0 : tensor<20x241xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = math.exp %a : f32
      linalg.yield %v : f32
  } -> tensor<20x241xf32>
  %si = linalg.fill ins(%zero : f32) outs(%re : tensor<20xf32>) -> tensor<20xf32>
  %s = linalg.reduce ins(%e : tensor<20x241xf32>) outs(%si : tensor<20xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.addf %in, %init : f32
      linalg.yield %v : f32
    }
  %se = tensor.expand_shape %s [[0, 1]] output_shape [20, 1] : tensor<20xf32> into tensor<20x1xf32>
  %y = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%e, %se : tensor<20x241xf32>, tensor<20x1xf32>) outs(%e0 : tensor<20x241xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.divf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x241xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<20x241xf32>, memref<20x241xf32>) -> ()
  return
}
