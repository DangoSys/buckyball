// The layer normalization of the frontend's decomposition over the rows of
// [20, 768] with gamma and beta: convert-linalg-to-hart replaces it by a
// bbacc.custom "layernorm" on LayerNormBall, gamma, beta and eps in its
// parameter bank.
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#col = affine_map<(d0, d1) -> (0, d1)>
#one = affine_map<(d0, d1) -> (0, 0)>
func.func @compute(%x: memref<20x768xf32>, %g: memref<768xf32>, %bt: memref<768xf32>, %out: memref<20x768xf32>) attributes {llvm.emit_c_interface} {
  %xt = bufferization.to_tensor %x restrict : memref<20x768xf32> to tensor<20x768xf32>
  %gt = bufferization.to_tensor %g restrict : memref<768xf32> to tensor<768xf32>
  %bb = bufferization.to_tensor %bt restrict : memref<768xf32> to tensor<768xf32>
  %zero = arith.constant 0.0 : f32
  %inv = arith.constant dense<0.00130208337> : tensor<1x1xf32>
  %eps = arith.constant dense<9.99999997E-7> : tensor<20x1xf32>
  %re = tensor.empty() : tensor<20xf32>
  %ri = linalg.fill ins(%zero : f32) outs(%re : tensor<20xf32>) -> tensor<20xf32>
  %e1 = tensor.empty() : tensor<20x1xf32>
  %e0 = tensor.empty() : tensor<20x768xf32>
  %s = linalg.reduce ins(%xt : tensor<20x768xf32>) outs(%ri : tensor<20xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.addf %in, %init : f32
      linalg.yield %v : f32
    }
  %se = tensor.expand_shape %s [[0, 1]] output_shape [20, 1] : tensor<20xf32> into tensor<20x1xf32>
  %mean = linalg.generic {indexing_maps = [#one, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%inv, %se : tensor<1x1xf32>, tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %d = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%xt, %mean : tensor<20x768xf32>, tensor<20x1xf32>) outs(%e0 : tensor<20x768xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.subf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x768xf32>
  %sq = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%d, %d : tensor<20x768xf32>, tensor<20x768xf32>) outs(%e0 : tensor<20x768xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x768xf32>
  %q = linalg.reduce ins(%sq : tensor<20x768xf32>) outs(%ri : tensor<20xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.addf %in, %init : f32
      linalg.yield %v : f32
    }
  %qe = tensor.expand_shape %q [[0, 1]] output_shape [20, 1] : tensor<20xf32> into tensor<20x1xf32>
  %var = linalg.generic {indexing_maps = [#one, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%inv, %qe : tensor<1x1xf32>, tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %ve = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%var, %eps : tensor<20x1xf32>, tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ve : tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = math.rsqrt %a : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %n = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%d, %r : tensor<20x768xf32>, tensor<20x1xf32>) outs(%e0 : tensor<20x768xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x768xf32>
  %ge = tensor.expand_shape %gt [[0, 1]] output_shape [1, 768] : tensor<768xf32> into tensor<1x768xf32>
  %ng = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel", "parallel"]}
      ins(%n, %ge : tensor<20x768xf32>, tensor<1x768xf32>) outs(%e0 : tensor<20x768xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x768xf32>
  %be = tensor.expand_shape %bb [[0, 1]] output_shape [1, 768] : tensor<768xf32> into tensor<1x768xf32>
  %y = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ng, %be : tensor<20x768xf32>, tensor<1x768xf32>) outs(%e0 : tensor<20x768xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x768xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<20x768xf32>, memref<20x768xf32>) -> ()
  return
}
