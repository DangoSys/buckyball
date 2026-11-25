// BANK: bbacc.mset <bank = {{[0-9]+}}, alloc = true, row = 1, col = 4>
// BANK: mxint8gemv_ball.gemv_mxint8 <a = {{[0-9]+}}, b = {{[0-9]+}}, c = {{[0-9]+}}, k = 512, a_codes = 2, a_exponents = 40, b_base = 3, c_base = 4, accumulate = false, finalize = false>
// BANK: mxint8gemv_ball.gemv_mxint8 <a = {{[0-9]+}}, b = {{[0-9]+}}, c = {{[0-9]+}}, k = 512, a_codes = 2, a_exponents = 40, b_base = 3, c_base = 4, accumulate = true, finalize = true>
func.func @compute(%input: memref<528xi8>, %weights: memref<16896xi8>, %out: memref<336xi8>) attributes {llvm.emit_c_interface} {
  %z = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %seventeen = arith.constant 17 : index
  %thirtytwo = arith.constant 32 : index
  %forty = arith.constant 40 : index
  %rows = arith.constant 264 : index
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 4>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %codes = memref.subview %input[0] [512] [1] : memref<528xi8> to memref<512xi8, strided<[1]>>
  %scales = memref.subview %input[512] [16] [1] : memref<528xi8> to memref<16xi8, strided<[1], offset: 512>>
  %x = bbacc.bank_mvin_strided %codes, %a at %two rows %thirtytwo memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<512xi8, strided<[1]>>
  %y = bbacc.bank_mvin_strided %scales, %x at %forty rows %c1 memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<16xi8, strided<[1], offset: 512>>
  %w0 = memref.subview %weights[0] [4224] [1] : memref<16896xi8> to memref<4224xi8, strided<[1]>>
  %w1 = memref.subview %weights[4224] [4224] [1] : memref<16896xi8> to memref<4224xi8, strided<[1], offset: 4224>>
  %w2 = memref.subview %weights[8448] [4224] [1] : memref<16896xi8> to memref<4224xi8, strided<[1], offset: 8448>>
  %w3 = memref.subview %weights[12672] [4224] [1] : memref<16896xi8> to memref<4224xi8, strided<[1], offset: 12672>>
  %b0 = bbacc.bank_mvin_strided %w0, %b at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<4224xi8, strided<[1]>>
  %b1 = bbacc.bank_mvin_strided %w1, %b0 at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 1, bank_stride = 1> : memref<4224xi8, strided<[1], offset: 4224>>
  %b2 = bbacc.bank_mvin_strided %w2, %b1 at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 2, bank_stride = 1> : memref<4224xi8, strided<[1], offset: 8448>>
  %b3 = bbacc.bank_mvin_strided %w3, %b2 at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 3, bank_stride = 1> : memref<4224xi8, strided<[1], offset: 12672>>
  %state = mxint8gemv_ball.bank_gemv_mxint8 %y, %b3 -> %c <k = 512, a_codes = 2, a_exponents = 40, b_base = 3, c_base = 4, accumulate = false, finalize = false>
  %partial = memref.subview %out[0] [272] [1] : memref<336xi8> to memref<272xi8, strided<[1]>>
  %stored = bbacc.bank_mvout_strided %partial, %state at %four rows %seventeen memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<272xi8, strided<[1]>>
  %result = mxint8gemv_ball.bank_gemv_mxint8 %y, %b3 -> %stored <k = 512, a_codes = 2, a_exponents = 40, b_base = 3, c_base = 4, accumulate = true, finalize = true>
  %final = memref.subview %out[272] [48] [1] : memref<336xi8> to memref<48xi8, strided<[1], offset: 272>>
  %done = bbacc.bank_mvout_strided %final, %result at %four rows %three memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<48xi8, strided<[1], offset: 272>>
  bbacc.bank_release %y
  bbacc.bank_release %b3
  bbacc.bank_release %done
  %verify = bbacc.bank_alloc <row = 1, col = 1>
  %readback = bbacc.bank_mvin %out, %verify <depth = 21, stride = 1> : memref<336xi8>
  bbacc.bank_release %readback
  return
}
