// HART-LABEL: func.func private @activation_destination_ht0(
// HART-SAME: %arg2: tensor<1x132xi8> {{.*}}bufferization.writable = true
// BUFFER-LABEL: func.func private @activation_destination_ht0(
// BUFFER-NOT: memref.alloc
// BUFFER-NOT: memref.copy
// BUFFER: %[[OUTPUT:.*]] = memref.collapse_shape %arg2
// BUFFER-NOT: memref.alloc
// BUFFER-NOT: memref.copy
// BUFFER: %[[CODES_VIEW:.*]] = memref.subview %[[OUTPUT]]
// BUFFER: %[[CODES:.*]] = memref.cast %[[CODES_VIEW]]
// BUFFER: bbacc.bank_mvout_strided %[[CODES]],
// BUFFER: %[[SCALES_VIEW:.*]] = memref.subview %[[OUTPUT]]
// BUFFER: %[[SCALES:.*]] = memref.cast %[[SCALES_VIEW]]
// BUFFER: bbacc.bank_mvout_strided %[[SCALES]],
// BUFFER-NOT: memref.alloc
// BUFFER-NOT: memref.copy
// BUFFER: return
func.func @activation_destination(%input: tensor<1x132xi8>, %table: tensor<4096xi32>, %out: tensor<1x132xi8>) -> tensor<1x132xi8> {
  %result = "frontend.mxint8_activation"(%input, %input, %table) {axis = 1 : i64, gated = false} : (tensor<1x132xi8>, tensor<1x132xi8>, tensor<4096xi32>) -> tensor<1x132xi8>
  %written = bufferization.materialize_in_destination %result in %out : (tensor<1x132xi8>, tensor<1x132xi8>) -> tensor<1x132xi8>
  return %written : tensor<1x132xi8>
}
func.func @compute_activation_destination(%input: memref<1x132xi8, strided<[132,1], offset: ?>>, %table: memref<4096xi32>, %out: memref<1x132xi8, strided<[132,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<1x132xi8, strided<[132,1], offset: ?>> to tensor<1x132xi8>
  %t = bufferization.to_tensor %table restrict : memref<4096xi32> to tensor<4096xi32>
  %o = bufferization.to_tensor %out restrict writable : memref<1x132xi8, strided<[132,1], offset: ?>> to tensor<1x132xi8>
  %result = call @activation_destination(%x, %t, %o) : (tensor<1x132xi8>, tensor<4096xi32>, tensor<1x132xi8>) -> tensor<1x132xi8>
  bufferization.materialize_in_destination %result in restrict writable %out : (tensor<1x132xi8>, memref<1x132xi8, strided<[132,1], offset: ?>>) -> ()
  return
}
