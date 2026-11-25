func.func @compute(%input: memref<8x48xi8>, %gated: memref<16x48xi8>, %table: memref<4096xi32>, %output: memref<2x8x48xi8>) attributes {llvm.emit_c_interface} {
  %a = memref.subview %input[0, 0] [8, 33] [1, 1] : memref<8x48xi8> to memref<8x33xi8, strided<[48, 1]>>
  %g = memref.subview %gated[8, 0] [8, 33] [1, 1] : memref<16x48xi8> to memref<8x33xi8, strided<[48, 1], offset: 384>>
  %at = bufferization.to_tensor %a restrict : memref<8x33xi8, strided<[48, 1]>> to tensor<8x33xi8>
  %gt = bufferization.to_tensor %g restrict : memref<8x33xi8, strided<[48, 1], offset: 384>> to tensor<8x33xi8>
  %tt = bufferization.to_tensor %table restrict : memref<4096xi32> to tensor<4096xi32>
  %x = "frontend.mxint8_activation"(%at, %at, %tt) {axis = 1 : i64, gated = false} : (tensor<8x33xi8>, tensor<8x33xi8>, tensor<4096xi32>) -> tensor<8x33xi8>
  %y = "frontend.mxint8_activation"(%at, %gt, %tt) {axis = 1 : i64, gated = true} : (tensor<8x33xi8>, tensor<8x33xi8>, tensor<4096xi32>) -> tensor<8x33xi8>
  %out0 = memref.subview %output[0, 0, 0] [1, 8, 33] [1, 1, 1] : memref<2x8x48xi8> to memref<8x33xi8, strided<[48, 1]>>
  %out1 = memref.subview %output[1, 0, 0] [1, 8, 33] [1, 1, 1] : memref<2x8x48xi8> to memref<8x33xi8, strided<[48, 1], offset: 384>>
  bufferization.materialize_in_destination %x in restrict writable %out0 : (tensor<8x33xi8>, memref<8x33xi8, strided<[48, 1]>>) -> ()
  bufferization.materialize_in_destination %y in restrict writable %out1 : (tensor<8x33xi8>, memref<8x33xi8, strided<[48, 1], offset: 384>>) -> ()
  %flat = memref.collapse_shape %output [[0, 1, 2]] : memref<2x8x48xi8> into memref<768xi8>
  %bank = bbacc.bank_alloc <row = 1, col = 1>
  %readback = bbacc.bank_mvin %flat, %bank <depth = 48, stride = 1> : memref<768xi8>
  bbacc.bank_release %readback
  return
}
