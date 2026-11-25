// POW-LABEL: func.func @compute(
// POW: math.powf
// POW: -> tensor<?x1xf32>
// POW: hart.launch
// POW-NOT: math.powf
// POW: return
// CHECK: bbacc.bank_run_operator
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#column = affine_map<(d0, d1) -> (d1)>
func.func @compute(%input: memref<?x?xf32>, %scales: memref<?x1xf32>, %weights: memref<?xf32>, %out: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<?x?xf32> to tensor<?x?xf32>
  %s = bufferization.to_tensor %scales restrict : memref<?x1xf32> to tensor<?x1xf32>
  %w = bufferization.to_tensor %weights restrict : memref<?xf32> to tensor<?xf32>
  %zero = arith.constant 0 : index
  %rows = tensor.dim %a, %zero : tensor<?x?xf32>
  %one = arith.constant 1 : index
  %columns = tensor.dim %a, %one : tensor<?x?xf32>
  %power_empty = tensor.empty(%rows) : tensor<?x1xf32>
  %exponent = arith.constant -0.5 : f32
  %normalized = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%s : tensor<?x1xf32>) outs(%power_empty : tensor<?x1xf32>) {
  ^bb0(%x: f32, %unused: f32):
    %value = math.powf %x, %exponent : f32
    linalg.yield %value : f32
  } -> tensor<?x1xf32>
  %empty = tensor.empty(%rows, %columns) : tensor<?x?xf32>
  %scaled = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %normalized : tensor<?x?xf32>, tensor<?x1xf32>) outs(%empty : tensor<?x?xf32>) {
  ^bb0(%x: f32, %scale: f32, %unused: f32):
    %product = arith.mulf %x, %scale : f32
    linalg.yield %product : f32
  } -> tensor<?x?xf32>
  %result_rows = tensor.dim %scaled, %zero : tensor<?x?xf32>
  %result_columns = tensor.dim %scaled, %one : tensor<?x?xf32>
  %result_empty = tensor.empty(%result_rows, %result_columns) : tensor<?x?xf32>
  %value = linalg.generic {indexing_maps = [#id, #column, #id], iterator_types = ["parallel", "parallel"]} ins(%scaled, %w : tensor<?x?xf32>, tensor<?xf32>) outs(%result_empty : tensor<?x?xf32>) {
  ^bb0(%x: f32, %weight: f32, %unused: f32):
    %sum = arith.addf %x, %weight : f32
    linalg.yield %sum : f32
  } -> tensor<?x?xf32>
  bufferization.materialize_in_destination %value in writable %out : (tensor<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

#cube = affine_map<(h,m,n)->(h,m,n)>
func.func @compute_strided(%a: memref<2x?x32xf32, strided<[?,32,1], offset: ?>>, %out: memref<2x?x32xf32, strided<[?,32,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %a restrict : memref<2x?x32xf32, strided<[?,32,1], offset: ?>> to tensor<2x?x32xf32>
  %one = arith.constant 1 : index
  %rows = tensor.dim %x, %one : tensor<2x?x32xf32>
  %empty = tensor.empty(%rows) : tensor<2x?x32xf32>
  %value = linalg.generic {indexing_maps=[#cube,#cube],iterator_types=["parallel","parallel","parallel"]}
      ins(%x : tensor<2x?x32xf32>) outs(%empty : tensor<2x?x32xf32>) {
  ^bb0(%v:f32,%unused:f32):
    %sum = arith.addf %v,%v : f32
    linalg.yield %sum : f32
  } -> tensor<2x?x32xf32>
  %expanded = tensor.expand_shape %value [[0], [1], [2,3]] output_shape [2, %rows, 1, 32] : tensor<2x?x32xf32> into tensor<2x?x1x32xf32>
  %target = tensor.empty(%rows) : tensor<2x1x?x32xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x?x1x32xf32>) outs(%target : tensor<2x1x?x32xf32>) permutation = [0,2,1,3]
  %view = tensor.collapse_shape %transposed [[0,1],[2],[3]] : tensor<2x1x?x32xf32> into tensor<2x?x32xf32>
  bufferization.materialize_in_destination %view in writable %out : (tensor<2x?x32xf32>,memref<2x?x32xf32, strided<[?,32,1], offset: ?>>) -> ()
  return
}

#pipeline_id = affine_map<(m,n)->(m,n)>
func.func private @pipeline(%a: tensor<?x33xi8> {bufferization.buffer_layout = strided<[33,1], offset: ?>}, %w: tensor<32x33xi8> {bufferization.buffer_layout = strided<[33,1], offset: ?>}, %out: tensor<?x32xf32> {bufferization.writable = true}) -> tensor<?x32xf32> {
  %x = "frontend.mxint8_to_fp32"(%a) {axis = 1 : i64} : (tensor<?x33xi8>) -> tensor<?x32xf32>
  %y = "frontend.mxint8_to_fp32"(%w) {axis = 1 : i64} : (tensor<32x33xi8>) -> tensor<32x32xf32>
  %weight_empty = tensor.empty() : tensor<32x32xf32>
  %weights = linalg.transpose ins(%y : tensor<32x32xf32>) outs(%weight_empty : tensor<32x32xf32>) permutation = [1,0]
  %zero = arith.constant 0 : index
  %rows = tensor.dim %a, %zero : tensor<?x33xi8>
  %empty = tensor.empty(%rows) : tensor<?x32xf32>
  %z = arith.constant 0.0 : f32
  %init = linalg.fill ins(%z : f32) outs(%empty : tensor<?x32xf32>) -> tensor<?x32xf32>
  %first = linalg.matmul ins(%x,%weights : tensor<?x32xf32>,tensor<32x32xf32>) outs(%init : tensor<?x32xf32>) -> tensor<?x32xf32>
  %neg = linalg.generic {indexing_maps=[#pipeline_id,#pipeline_id],iterator_types=["parallel","parallel"]}
      ins(%first : tensor<?x32xf32>) outs(%empty : tensor<?x32xf32>) {
  ^bb0(%v:f32,%unused:f32):
    %negative = math.erf %v fastmath<afn> : f32
    linalg.yield %negative : f32
  } -> tensor<?x32xf32>
  %q = "frontend.fp32_to_mxint8"(%neg) {axis = 1 : i64} : (tensor<?x32xf32>) -> tensor<?x33xi8>
  %d = "frontend.mxint8_to_fp32"(%q) {axis = 1 : i64} : (tensor<?x33xi8>) -> tensor<?x32xf32>
  %second = linalg.matmul ins(%d,%weights : tensor<?x32xf32>,tensor<32x32xf32>) outs(%init : tensor<?x32xf32>) -> tensor<?x32xf32>
  %written = bufferization.materialize_in_destination %second in %out : (tensor<?x32xf32>,tensor<?x32xf32>) -> tensor<?x32xf32>
  return %written : tensor<?x32xf32>
}
func.func @compute_pipeline(%a: memref<?x33xi8, strided<[33,1], offset: ?>>, %w: memref<32x33xi8, strided<[33,1], offset: ?>>, %out: memref<?x32xf32, strided<[32,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %a restrict : memref<?x33xi8, strided<[33,1], offset: ?>> to tensor<?x33xi8>
  %y = bufferization.to_tensor %w restrict : memref<32x33xi8, strided<[33,1], offset: ?>> to tensor<32x33xi8>
  %target = bufferization.to_tensor %out restrict writable : memref<?x32xf32, strided<[32,1], offset: ?>> to tensor<?x32xf32>
  %value = func.call @pipeline(%x,%y,%target) : (tensor<?x33xi8>,tensor<32x33xi8>,tensor<?x32xf32>) -> tensor<?x32xf32>
  bufferization.materialize_in_destination %value in writable %out : (tensor<?x32xf32>,memref<?x32xf32, strided<[32,1], offset: ?>>) -> ()
  return
}
