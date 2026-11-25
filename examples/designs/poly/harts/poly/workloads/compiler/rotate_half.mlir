// HART-LABEL: func.func @concat_only
// HART: tensor.concat
// HART-LABEL: func.func @mismatched_extents
// HART: tensor.concat
// HART-LABEL: func.func @compute_rotate_half
// HART-NOT: tensor.concat
// HART-NOT: linalg.generic
// HART: hart.launch
// HART-NOT: tensor.concat
// HART-NOT: linalg.generic
// HART: return
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#coef = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#row = affine_map<(d0, d1) -> (d0, d1)>
func.func @concat_only(%x: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %x, %c0 : tensor<?x128xf32>
  %lo = tensor.extract_slice %x[0, 0] [%m, 64] [1, 1] : tensor<?x128xf32> to tensor<?x64xf32>
  %hi = tensor.extract_slice %x[0, 64] [%m, 64] [1, 1] : tensor<?x128xf32> to tensor<?x64xf32>
  %e = tensor.empty(%m) : tensor<?x64xf32>
  %negative = linalg.generic {indexing_maps = [#row, #row], iterator_types = ["parallel", "parallel"]} ins(%hi : tensor<?x64xf32>) outs(%e : tensor<?x64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %v = arith.negf %a : f32
    linalg.yield %v : f32
  } -> tensor<?x64xf32>
  %cat = tensor.concat dim(1) %negative, %lo : (tensor<?x64xf32>, tensor<?x64xf32>) -> tensor<?x128xf32>
  return %cat : tensor<?x128xf32>
}
func.func @mismatched_extents(%x: tensor<?x128xf32>, %n: index) -> tensor<?x128xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %x, %c0 : tensor<?x128xf32>
  %lo = tensor.extract_slice %x[0, 0] [%m, 64] [1, 1] : tensor<?x128xf32> to tensor<?x64xf32>
  %hi = tensor.extract_slice %x[0, 64] [%n, 64] [1, 1] : tensor<?x128xf32> to tensor<?x64xf32>
  %e = tensor.empty(%n) : tensor<?x64xf32>
  %negative = linalg.generic {indexing_maps = [#row, #row], iterator_types = ["parallel", "parallel"]} ins(%hi : tensor<?x64xf32>) outs(%e : tensor<?x64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %v = arith.negf %a : f32
    linalg.yield %v : f32
  } -> tensor<?x64xf32>
  %cat = tensor.concat dim(1) %negative, %lo : (tensor<?x64xf32>, tensor<?x64xf32>) -> tensor<?x128xf32>
  %empty = tensor.empty(%m) : tensor<?x128xf32>
  %result = linalg.generic {indexing_maps = [#row, #row, #row], iterator_types = ["parallel", "parallel"]} ins(%x, %cat : tensor<?x128xf32>, tensor<?x128xf32>) outs(%empty : tensor<?x128xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %v = arith.addf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<?x128xf32>
  return %result : tensor<?x128xf32>
}
func.func @compute_rotate_half(%x: tensor<1x4x?x128xf32>, %cos: tensor<1x1x?x128xf32>, %sin: tensor<1x1x?x128xf32>, %output: tensor<1x4x?x128xf32> {bufferization.writable = true}) -> tensor<1x4x?x128xf32> attributes {llvm.emit_c_interface} {
  %c2 = arith.constant 2 : index
  %m = tensor.dim %x, %c2 : tensor<1x4x?x128xf32>
  %lo = tensor.extract_slice %x[0, 0, 0, 0] [1, 4, %m, 64] [1, 1, 1, 1] : tensor<1x4x?x128xf32> to tensor<1x4x?x64xf32>
  %hi = tensor.extract_slice %x[0, 0, 0, 64] [1, 4, %m, 64] [1, 1, 1, 1] : tensor<1x4x?x128xf32> to tensor<1x4x?x64xf32>
  %half = tensor.empty(%m) : tensor<1x4x?x64xf32>
  %negative = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%hi : tensor<1x4x?x64xf32>) outs(%half : tensor<1x4x?x64xf32>) {
  ^bb0(%a: f32, %o: f32):
    %v = arith.negf %a : f32
    linalg.yield %v : f32
  } -> tensor<1x4x?x64xf32>
  %cat = tensor.concat dim(3) %negative, %lo : (tensor<1x4x?x64xf32>, tensor<1x4x?x64xf32>) -> tensor<1x4x?x128xf32>
  %empty = tensor.empty(%m) : tensor<1x4x?x128xf32>
  %first = linalg.generic {indexing_maps = [#id, #coef, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x, %cos : tensor<1x4x?x128xf32>, tensor<1x1x?x128xf32>) outs(%empty : tensor<1x4x?x128xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %v = arith.mulf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<1x4x?x128xf32>
  %second = linalg.generic {indexing_maps = [#id, #coef, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cat, %sin : tensor<1x4x?x128xf32>, tensor<1x1x?x128xf32>) outs(%empty : tensor<1x4x?x128xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %v = arith.mulf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<1x4x?x128xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%first, %second : tensor<1x4x?x128xf32>, tensor<1x4x?x128xf32>) outs(%empty : tensor<1x4x?x128xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %v = arith.addf %a, %b : f32
    linalg.yield %v : f32
  } -> tensor<1x4x?x128xf32>
  %written = bufferization.materialize_in_destination %result in %output : (tensor<1x4x?x128xf32>, tensor<1x4x?x128xf32>) -> tensor<1x4x?x128xf32>
  return %written : tensor<1x4x?x128xf32>
}
