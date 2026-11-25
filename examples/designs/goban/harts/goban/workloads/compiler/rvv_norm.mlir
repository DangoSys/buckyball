#id = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (0, d1)>
func.func @compute(%a: memref<2x36xf32>, %b: memref<1x36xf32>, %out: memref<2x36xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x36xf32> to tensor<2x36xf32>
  %bt = bufferization.to_tensor %b restrict : memref<1x36xf32> to tensor<1x36xf32>
  %empty = tensor.empty() : tensor<2x36xf32>
  %y = linalg.generic {indexing_maps = [#id, #bias, #id], iterator_types = ["parallel", "parallel"]}
      ins(%at, %bt : tensor<2x36xf32>, tensor<1x36xf32>) outs(%empty : tensor<2x36xf32>) {
  ^bb0(%av: f32, %bv: f32, %unused: f32):
    %epsilon = arith.constant 0.125 : f32
    %absolute = math.absf %av : f32
    %denominator = arith.addf %absolute, %epsilon : f32
    %inverse = math.rsqrt %denominator : f32
    %scaled = arith.mulf %av, %inverse : f32
    %result = arith.addf %scaled, %bv : f32
    linalg.yield %result : f32
  } -> tensor<2x36xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<2x36xf32>, memref<2x36xf32>) -> ()
  return
}
