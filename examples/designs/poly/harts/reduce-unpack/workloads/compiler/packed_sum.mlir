// FUSED-LABEL: func.func @compute(
// FUSED-NOT: memref.alloc
// FUSED-NOT: frontend.mxint8_to_fp32
// FUSED: mxint8unpack_ball.bank_unpack
// FUSED-NOT: bbacc.bank_mvout
// FUSED: mxint8unpack_ball.bank_unpack
// FUSED-NOT: bbacc.bank_mvout
// FUSED: f32add_ball.bank_add_f32
// FUSED: bbacc.bank_mvout
// FUSED-NOT: bbacc.bank_mvout
// FUSED-LABEL: func.func @shared(
// FUSED: mxint8unpack_ball.bank_unpack
// FUSED: bbacc.bank_mvout
// FUSED: mxint8unpack_ball.bank_unpack
// FUSED: bbacc.bank_mvout
// FUSED: bbacc.bank_mvin
// FUSED: f32add_ball.bank_add_f32
#id=affine_map<(i,j)->(i,j)>
module {
  func.func @compute(%p0: tensor<?x132xi8> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false}, %p1: tensor<?x132xi8> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false}, %out: tensor<?x128xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true}) -> tensor<?x128xf32> attributes {llvm.emit_c_interface} {
    %d0="frontend.mxint8_to_fp32"(%p0) {axis=1:i64} : (tensor<?x132xi8>)->tensor<?x128xf32>
    %d1="frontend.mxint8_to_fp32"(%p1) {axis=1:i64} : (tensor<?x132xi8>)->tensor<?x128xf32>
    %c0=arith.constant 0:index
    %rows=tensor.dim %d1,%c0:tensor<?x128xf32>
    %empty=tensor.empty(%rows):tensor<?x128xf32>
    %s0=linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel"]} ins(%d1,%d0:tensor<?x128xf32>,tensor<?x128xf32>) outs(%empty:tensor<?x128xf32>) {
    ^bb0(%a:f32,%b:f32,%unused:f32):
      %v=arith.addf %a,%b:f32
      linalg.yield %v:f32
    }->tensor<?x128xf32>
    %written0=bufferization.materialize_in_destination %s0 in %out : (tensor<?x128xf32>,tensor<?x128xf32>)->tensor<?x128xf32>
    return %written0 : tensor<?x128xf32>
  }
  func.func @shared(%p0: tensor<?x132xi8> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false}, %p1: tensor<?x132xi8> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false}, %out: tensor<?x128xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true}, %extra: tensor<?x128xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true}) -> (tensor<?x128xf32>, tensor<?x128xf32>) attributes {llvm.emit_c_interface} {
    %d0="frontend.mxint8_to_fp32"(%p0) {axis=1:i64} : (tensor<?x132xi8>)->tensor<?x128xf32>
    %d1="frontend.mxint8_to_fp32"(%p1) {axis=1:i64} : (tensor<?x132xi8>)->tensor<?x128xf32>
    %c0=arith.constant 0:index
    %rows=tensor.dim %d1,%c0:tensor<?x128xf32>
    %empty=tensor.empty(%rows):tensor<?x128xf32>
    %sum=linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel"]} ins(%d1,%d0:tensor<?x128xf32>,tensor<?x128xf32>) outs(%empty:tensor<?x128xf32>) {
    ^bb0(%a:f32,%b:f32,%unused:f32):
      %v=arith.addf %a,%b:f32
      linalg.yield %v:f32
    }->tensor<?x128xf32>
    %written0=bufferization.materialize_in_destination %sum in %out : (tensor<?x128xf32>,tensor<?x128xf32>)->tensor<?x128xf32>
    %written1=bufferization.materialize_in_destination %d1 in %extra : (tensor<?x128xf32>,tensor<?x128xf32>)->tensor<?x128xf32>
    return %written0, %written1 : tensor<?x128xf32>, tensor<?x128xf32>
  }
}
