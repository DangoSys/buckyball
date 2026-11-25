// CHECK: bbacc.bank_run_operator
// CHECK: arith.sitofp {{.*}} : vector<32xi8> to vector<32xf32>
// CHECK: arith.remui {{.*}} : index
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#row = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#column = affine_map<(d0, d1, d2) -> (d2)>
#rope = affine_map<(d0, d1) -> (d0, d1)>
#shift = affine_map<(d0, d1) -> (d0, (d1 + 32) mod 64)>
#coefficient = affine_map<(d0, d1) -> (d1)>
#block = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @compute(%input: memref<?x64xf32>, %scales: memref<?x1xf32>, %weights: memref<32xf32>, %codes: memref<?x64xi8>, %exponents: memref<?x2xi8>, %out: memref<?x32xf32>) attributes {llvm.emit_c_interface} {
  %full = bufferization.to_tensor %input restrict : memref<?x64xf32> to tensor<?x64xf32>
  %c0 = arith.constant 0 : index
  %height = tensor.dim %full, %c0 : tensor<?x64xf32>
  %raw_codes = bufferization.to_tensor %codes restrict : memref<?x64xi8> to tensor<?x64xi8>
  %blocks = tensor.expand_shape %raw_codes [[0], [1, 2]] output_shape [%height, 2, 32] : tensor<?x64xi8> into tensor<?x2x32xi8>
  %raw_exponents = bufferization.to_tensor %exponents restrict : memref<?x2xi8> to tensor<?x2xi8>
  %decode_empty = tensor.empty(%height) : tensor<?x2x32xf32>
  %c127 = arith.constant 127 : i32
  %c1 = arith.constant 1 : i32
  %c23 = arith.constant 23 : i32
  %decoded = linalg.generic {indexing_maps = [#id, #block, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%blocks, %raw_exponents : tensor<?x2x32xi8>, tensor<?x2xi8>) outs(%decode_empty : tensor<?x2x32xf32>) {
  ^bb0(%mantissa: i8, %exponent: i8, %unused: f32):
    %e = arith.extui %exponent : i8 to i32
    %delta = arith.subi %e, %c127 : i32
    %lo = arith.shrsi %delta, %c1 : i32
    %hi = arith.subi %delta, %lo : i32
    %a = arith.addi %lo, %c127 : i32
    %a_bits = arith.shli %a, %c23 : i32
    %a_scale = arith.bitcast %a_bits : i32 to f32
    %b = arith.addi %hi, %c127 : i32
    %b_bits = arith.shli %b, %c23 : i32
    %b_scale = arith.bitcast %b_bits : i32 to f32
    %value = arith.sitofp %mantissa : i8 to f32
    %first = arith.mulf %value, %a_scale : f32
    %result = arith.mulf %first, %b_scale : f32
    linalg.yield %result : f32
  } -> tensor<?x2x32xf32>
  %decoded_rows = tensor.collapse_shape %decoded [[0], [1, 2]] : tensor<?x2x32xf32> into tensor<?x64xf32>
  %residual_empty = tensor.empty(%height) : tensor<?x64xf32>
  %residual = linalg.generic {indexing_maps = [#rope, #rope, #rope], iterator_types = ["parallel", "parallel"]} ins(%full, %decoded_rows : tensor<?x64xf32>, tensor<?x64xf32>) outs(%residual_empty : tensor<?x64xf32>) {
  ^bb0(%x: f32, %restored: f32, %unused: f32):
    %difference = arith.subf %x, %restored : f32
    linalg.yield %difference : f32
  } -> tensor<?x64xf32>
  %cos = arith.constant dense<1.5> : tensor<64xf32>
  %sin = arith.constant dense<0.25> : tensor<64xf32>
  %limit = arith.constant 32 : index
  %rope_empty = tensor.empty(%height) : tensor<?x64xf32>
  %rotated = linalg.generic {indexing_maps = [#rope, #shift, #coefficient, #coefficient, #rope], iterator_types = ["parallel", "parallel"]} ins(%residual, %residual, %cos, %sin : tensor<?x64xf32>, tensor<?x64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%rope_empty : tensor<?x64xf32>) {
  ^bb0(%x: f32, %paired: f32, %cosine: f32, %sine: f32, %unused: f32):
    %column = linalg.index 1 : index
    %lower = arith.cmpi ult, %column, %limit : index
    %negative = arith.negf %paired : f32
    %signed = arith.select %lower, %negative, %paired : f32
    %first = arith.mulf %x, %cosine : f32
    %second = arith.mulf %signed, %sine : f32
    %sum = arith.addf %first, %second : f32
    linalg.yield %sum : f32
  } -> tensor<?x64xf32>
  %a = tensor.extract_slice %rotated[0, 32] [%height, 32] [1, 1] : tensor<?x64xf32> to tensor<?x32xf32>
  %s = bufferization.to_tensor %scales restrict : memref<?x1xf32> to tensor<?x1xf32>
  %w = bufferization.to_tensor %weights restrict : memref<32xf32> to tensor<32xf32>
  %zero = arith.constant 0 : index
  %rows = tensor.dim %a, %zero : tensor<?x32xf32>
  %three = arith.constant 3 : index
  %per_head = arith.divui %rows, %three : index
  %heads = tensor.expand_shape %a [[0, 1], [2]] output_shape [3, %per_head, 32] : tensor<?x32xf32> into tensor<3x?x32xf32>
  %head_scales = tensor.expand_shape %s [[0, 1], [2]] output_shape [3, %per_head, 1] : tensor<?x1xf32> into tensor<3x?x1xf32>
  %empty = tensor.empty(%per_head) : tensor<3x?x32xf32>
  %head_value = linalg.generic {indexing_maps = [#id, #row, #column, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%heads, %head_scales, %w : tensor<3x?x32xf32>, tensor<3x?x1xf32>, tensor<32xf32>) outs(%empty : tensor<3x?x32xf32>) {
  ^bb0(%x: f32, %scale: f32, %weight: f32, %unused: f32):
    %scaled = arith.mulf %x, %scale : f32
    %sum = arith.addf %scaled, %weight : f32
    linalg.yield %sum : f32
  } -> tensor<3x?x32xf32>
  %value = tensor.collapse_shape %head_value [[0, 1], [2]] : tensor<3x?x32xf32> into tensor<?x32xf32>
  bufferization.materialize_in_destination %value in writable %out : (tensor<?x32xf32>, memref<?x32xf32>) -> ()
  return
}
