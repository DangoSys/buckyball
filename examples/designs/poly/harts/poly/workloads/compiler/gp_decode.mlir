// CHECK: bbacc.bank_run_operator
// DECODE-LABEL: func.func @compute_decode(
// DECODE-NOT: frontend.mxint8_to_fp32
// DECODE: hart.launch @compute_decode_ht0
// DECODE: hart.launch @compute_decode_ht{{[1-9][0-9]*}}
// DECODE-NOT: frontend.mxint8_to_fp32
// DECODE: return
// DECODE-LABEL: func.func private @compute_decode_ht0(
// DECODE: bbacc.bank_run_operator
#id = affine_map<(d0,d1)->(d0,d1)>
func.func @compute_decode(%input: memref<?x132xi8, strided<[137,1], offset: ?>>, %out: memref<?x128xf32, strided<[132,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<?x132xi8, strided<[137,1], offset: ?>> to tensor<?x132xi8>
  %decoded = "frontend.mxint8_to_fp32"(%x) {axis = 1 : i64} : (tensor<?x132xi8>) -> tensor<?x128xf32>
  %zero = arith.constant 0 : index
  %rows = tensor.dim %x, %zero : tensor<?x132xi8>
  %empty = tensor.empty(%rows) : tensor<?x128xf32>
  %value = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel"]}
      ins(%decoded : tensor<?x128xf32>) outs(%empty : tensor<?x128xf32>) {
  ^bb0(%v:f32,%unused:f32):
    %negative = arith.negf %v : f32
    linalg.yield %negative : f32
  } -> tensor<?x128xf32>
  bufferization.materialize_in_destination %value in writable %out : (tensor<?x128xf32>,memref<?x128xf32, strided<[132,1], offset: ?>>) -> ()
  return
}
