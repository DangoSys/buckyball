func.func @compute(%input: memref<8x48xi8>, %gated: memref<16x48xi8>, %table: memref<4096xi32>, %output: memref<2x8x48xi8>) attributes {llvm.emit_c_interface} {
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 1>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %t = bbacc.bank_mvin %table, %b <depth = 1024, stride = 1> : memref<4096xi32>
  %x = bbacc.bank_mvin %input, %a <depth = 24, stride = 1> : memref<8x48xi8>
  %out0 = memref.subview %output[0, 0, 0] [1, 8, 48] [1, 1, 1] : memref<2x8x48xi8> to memref<1x8x48xi8, strided<[384, 48, 1]>>
  %y = mxint8act_ball.bank_act_mxint8 %x, %t -> %c <rows = 8, k = 32, in_base = 0, out_base = 0, table_base = 0, gated = false>
  %stored = bbacc.bank_mvout %out0, %y <depth = 24, stride = 1> : memref<1x8x48xi8, strided<[384, 48, 1]>>
  %g = bbacc.bank_mvin %gated, %x <depth = 48, stride = 1> : memref<16x48xi8>
  %out1 = memref.subview %output[1, 0, 0] [1, 8, 48] [1, 1, 1] : memref<2x8x48xi8> to memref<1x8x48xi8, strided<[384, 48, 1], offset: 384>>
  %z = mxint8act_ball.bank_act_mxint8 %g, %t -> %stored <rows = 8, k = 32, in_base = 0, out_base = 0, table_base = 0, gated = true>
  %done = bbacc.bank_mvout %out1, %z <depth = 24, stride = 1> : memref<1x8x48xi8, strided<[384, 48, 1], offset: 384>>
  bbacc.bank_release %g
  bbacc.bank_release %t
  bbacc.bank_release %done
  %flat = memref.collapse_shape %output [[0, 1, 2]] : memref<2x8x48xi8> into memref<768xi8>
  %verify = bbacc.bank_alloc <row = 1, col = 1>
  %readback = bbacc.bank_mvin %flat, %verify <depth = 48, stride = 1> : memref<768xi8>
  bbacc.bank_release %readback
  return
}
