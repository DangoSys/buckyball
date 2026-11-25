// BANK: scf.for
// BANK: mxint5gemv_ball.gemv_mxint5 {{.*}}update = true
// BANK: scf.for
// BANK: mxint5gemv_ball.gemv_mxint5 {{.*}}update = true
func.func @compute(%input: memref<528xi8>, %weights: memref<11264xi8>, %out: memref<1552xi8>) attributes {llvm.emit_c_interface} {
  %base = arith.constant 4 : i64
  %z = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %seventeen = arith.constant 17 : index
  %thirtytwo = arith.constant 32 : index
  %forty = arith.constant 40 : index
  %rows = arith.constant 176 : index
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 4>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %codes = memref.subview %input[0] [512] [1] : memref<528xi8> to memref<512xi8, strided<[1]>>
  %scales = memref.subview %input[512] [16] [1] : memref<528xi8> to memref<16xi8, strided<[1], offset: 512>>
  %x = bbacc.bank_mvin_strided %codes, %a at %two rows %thirtytwo memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<512xi8, strided<[1]>>
  %y = bbacc.bank_mvin_strided %scales, %x at %forty rows %c1 memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<16xi8, strided<[1], offset: 512>>
  %w0 = memref.subview %weights[0] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1]>>
  %w1 = memref.subview %weights[2816] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1], offset: 2816>>
  %w2 = memref.subview %weights[5632] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1], offset: 5632>>
  %w3 = memref.subview %weights[8448] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1], offset: 8448>>
  %b0 = bbacc.bank_mvin_strided %w0, %b at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<2816xi8, strided<[1]>>
  %b1 = bbacc.bank_mvin_strided %w1, %b0 at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 1, bank_stride = 1> : memref<2816xi8, strided<[1], offset: 2816>>
  %b2 = bbacc.bank_mvin_strided %w2, %b1 at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 2, bank_stride = 1> : memref<2816xi8, strided<[1], offset: 5632>>
  %b3 = bbacc.bank_mvin_strided %w3, %b2 at %three rows %rows memory_stride %sixteen bytes %sixteen block_log2 %z <group = 3, bank_stride = 1> : memref<2816xi8, strided<[1], offset: 8448>>
  %c32i = arith.constant 32 : index
  %c17i = arith.constant 17 : index
  %c48i = arith.constant 48 : index
  %c17v = arith.constant 17 : i64
  %all = scf.for %row = %z to %c32i step %c1 iter_args(%slot = %c) -> (i64) {
    %r = arith.muli %row, %c17i : index
    %at = arith.index_cast %r : index to i64
    %next = mxint5gemv_ball.bank_gemv_mxint5 %y, %b3 -> %slot at %at <k = 512, a_codes = 2, a_exponents = 40, b_base = 3, accumulate = false, finalize = false, update = true>
    scf.yield %next : i64
  }
  %done = scf.for %row = %z to %c32i step %c1 iter_args(%slot = %all) -> (i64) {
    %r = arith.muli %row, %c17i : index
    %at = arith.index_cast %r : index to i64
    %next = mxint5gemv_ball.bank_gemv_mxint5 %y, %b3 -> %slot at %at <k = 512, a_codes = 2, a_exponents = 40, b_base = 3, accumulate = true, finalize = true, update = true>
    %bankat = arith.muli %row, %c17i : index
    %memoryat = arith.muli %row, %c48i : index
    %final = memref.subview %out[%memoryat] [48] [1] : memref<1552xi8> to memref<48xi8, strided<[1], offset: ?>>
    %stored = bbacc.bank_mvout_strided %final, %next at %bankat rows %three memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<48xi8, strided<[1], offset: ?>>
    scf.yield %stored : i64
  }
  bbacc.bank_release %y
  bbacc.bank_release %b3
  bbacc.bank_release %done
  %verify = bbacc.bank_alloc <row = 1, col = 1>
  %readback = bbacc.bank_mvin %out, %verify <depth = 97, stride = 1> : memref<1552xi8>
  bbacc.bank_release %readback
  return
}
