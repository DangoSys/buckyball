#mixed_rows = affine_map<(m,n)->(m,n)>
// BATCH-LABEL: func.func @compute_mixed(
// BATCH-NOT: memref.copy
// BATCH: %[[COUNT:.*]] = arith.constant 4 : i64
// BATCH: call @stream_begin(%[[COUNT]],
// BATCH-NOT: call @stream_wait
// BATCH: call @mixed_ht3_launch
// BATCH: call @stream_submit()
// BATCH-NEXT: call @stream_wait()
// BATCH-NOT: call @stream_begin
// BATCH-NOT: memref.copy
// BATCH: return
// CHECK-LABEL: func.func private @mixed(
// CHECK: hart.launch @mixed_ht0{{.*}}thread = 0
// CHECK: hart.launch @mixed_ht1{{.*}}thread = 1
// CHECK: hart.launch @mixed_ht2{{.*}}thread = 0
// CHECK: hart.launch @mixed_ht3{{.*}}thread = 1
// CHECK-NOT: hart.launch
// CHECK: return
func.func private @mixed(%x: tensor<15x32xf32>, %narrow: tensor<15x32xf32> {bufferization.writable = true}, %wide: tensor<15x32xf32> {bufferization.writable = true}) -> (tensor<15x32xf32>, tensor<15x32xf32>) {
  %a = tensor.empty() : tensor<15x32xf32>
  %n = linalg.generic {indexing_maps=[#mixed_rows,#mixed_rows],iterator_types=["parallel","parallel"]}
      ins(%x : tensor<15x32xf32>) outs(%a : tensor<15x32xf32>) {
  ^bb0(%v:f32,%unused:f32):
    %sum = arith.addf %v,%v : f32
    linalg.yield %sum : f32
  } -> tensor<15x32xf32>
  %b = tensor.empty() : tensor<15x32xf32>
  %w = linalg.generic {indexing_maps=[#mixed_rows,#mixed_rows],iterator_types=["parallel","parallel"]}
      ins(%x : tensor<15x32xf32>) outs(%b : tensor<15x32xf32>) {
  ^bb0(%v:f32,%unused:f32):
    %value = arith.mulf %v, %v : f32
    linalg.yield %value : f32
  } -> tensor<15x32xf32>
  %n_written = bufferization.materialize_in_destination %n in %narrow : (tensor<15x32xf32>,tensor<15x32xf32>) -> tensor<15x32xf32>
  %w_written = bufferization.materialize_in_destination %w in %wide : (tensor<15x32xf32>,tensor<15x32xf32>) -> tensor<15x32xf32>
  return %n_written, %w_written : tensor<15x32xf32>, tensor<15x32xf32>
}
func.func @compute_mixed(%input: memref<15x32xf32, strided<[32,1], offset: ?>>, %narrow: memref<15x32xf32, strided<[32,1], offset: ?>>, %wide: memref<15x32xf32, strided<[32,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<15x32xf32, strided<[32,1], offset: ?>> to tensor<15x32xf32>
  %n = bufferization.to_tensor %narrow restrict writable : memref<15x32xf32, strided<[32,1], offset: ?>> to tensor<15x32xf32>
  %w = bufferization.to_tensor %wide restrict writable : memref<15x32xf32, strided<[32,1], offset: ?>> to tensor<15x32xf32>
  %out:2 = func.call @mixed(%x, %n, %w) : (tensor<15x32xf32>,tensor<15x32xf32>,tensor<15x32xf32>) -> (tensor<15x32xf32>,tensor<15x32xf32>)
  bufferization.materialize_in_destination %out#0 in writable %narrow : (tensor<15x32xf32>,memref<15x32xf32, strided<[32,1], offset: ?>>) -> ()
  bufferization.materialize_in_destination %out#1 in writable %wide : (tensor<15x32xf32>,memref<15x32xf32, strided<[32,1], offset: ?>>) -> ()
  return
}

// CHECK-LABEL: func.func private @quantized_destination_ht0(
// CHECK-SAME: %arg1: tensor<3x132xi8> {{.*}}bufferization.writable = true
func.func @quantized_destination(%input: tensor<3x128xf32>, %out: tensor<3x132xi8>) -> tensor<3x132xi8> {
  %q = "frontend.fp32_to_mxint8"(%input) {axis = 1 : i64} : (tensor<3x128xf32>) -> tensor<3x132xi8>
  %written = bufferization.materialize_in_destination %q in %out : (tensor<3x132xi8>, tensor<3x132xi8>) -> tensor<3x132xi8>
  return %written : tensor<3x132xi8>
}
