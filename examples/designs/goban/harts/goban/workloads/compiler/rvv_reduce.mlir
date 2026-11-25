#input = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#output = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @compute(%input: memref<16x1x36xf32>, %output: memref<16x1xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<16x1x36xf32> to tensor<16x1x36xf32>
  %batched = tensor.expand_shape %a [[0, 1], [2], [3]] output_shape [2, 8, 1, 36] : tensor<16x1x36xf32> into tensor<2x8x1x36xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x8x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x8x1xf32>) -> tensor<2x8x1xf32>
  %result = linalg.generic {indexing_maps = [#input, #output], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%batched : tensor<2x8x1x36xf32>) outs(%init : tensor<2x8x1xf32>) {
  ^bb0(%value: f32, %sum: f32):
    %next = arith.addf %value, %sum : f32
    linalg.yield %next : f32
  } -> tensor<2x8x1xf32>
  %flat = tensor.collapse_shape %result [[0, 1], [2]] : tensor<2x8x1xf32> into tensor<16x1xf32>
  bufferization.materialize_in_destination %flat in writable %output : (tensor<16x1xf32>, memref<16x1xf32>) -> ()
  return
}
