// LOWER-DAG: arith.constant 34362886145 : i64
// LOWER-DAG: arith.constant 34899757057 : i64
// LOWER-DAG: arith.constant 17592992006146 : i64
// LOWER-DAG: arith.constant 216190375105789954 : i64
// LOWER-DAG: llvm.mlir.constant([[FUNCT]] : i32) : i32
func.func @compute(%input: memref<48xi8>, %weights: memref<3072xi8>, %out: memref<336xi8>) attributes {llvm.emit_c_interface} {
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %seventeen = arith.constant 17 : index
  %twentyfour = arith.constant 24 : index
  %forty = arith.constant 40 : index
  %c384 = arith.constant 384 : index
  %c768 = arith.constant 768 : index
  %c1152 = arith.constant 1152 : index
  %c1536 = arith.constant 1536 : index
  %c272 = arith.constant 272 : index
  bbacc.mset <bank = 1, alloc = true, row = 1, col = 1>
  bbacc.mset <bank = 2, alloc = true, row = 1, col = 4>
  bbacc.mset <bank = 3, alloc = true, row = 1, col = 1>
  %codes = memref.subview %input[0] [32] [1] : memref<48xi8> to memref<32xi8, strided<[1]>>
  %scales = memref.subview %input[32] [16] [1] : memref<48xi8> to memref<16xi8, strided<[1], offset: 32>>
  bbacc.mvin_strided %codes, 1 at %two rows %two memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<32xi8, strided<[1]>>
  bbacc.mvin_strided %scales, 1 at %forty rows %one memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<16xi8, strided<[1], offset: 32>>
  scf.for %t = %z to %two step %one {
    %base = arith.muli %t, %c1536 : index
    %o1 = arith.addi %base, %c384 : index
    %o2 = arith.addi %base, %c768 : index
    %o3 = arith.addi %base, %c1152 : index
    %w0 = memref.subview %weights[%base] [384] [1] : memref<3072xi8> to memref<384xi8, strided<[1], offset: ?>>
    %w1 = memref.subview %weights[%o1] [384] [1] : memref<3072xi8> to memref<384xi8, strided<[1], offset: ?>>
    %w2 = memref.subview %weights[%o2] [384] [1] : memref<3072xi8> to memref<384xi8, strided<[1], offset: ?>>
    %w3 = memref.subview %weights[%o3] [384] [1] : memref<3072xi8> to memref<384xi8, strided<[1], offset: ?>>
    bbacc.mvin_strided %w0, 2 at %three rows %twentyfour memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<384xi8, strided<[1], offset: ?>>
    bbacc.mvin_strided %w1, 2 at %three rows %twentyfour memory_stride %sixteen bytes %sixteen block_log2 %z <group = 1, bank_stride = 1> : memref<384xi8, strided<[1], offset: ?>>
    bbacc.mvin_strided %w2, 2 at %three rows %twentyfour memory_stride %sixteen bytes %sixteen block_log2 %z <group = 2, bank_stride = 1> : memref<384xi8, strided<[1], offset: ?>>
    bbacc.mvin_strided %w3, 2 at %three rows %twentyfour memory_stride %sixteen bytes %sixteen block_log2 %z <group = 3, bank_stride = 1> : memref<384xi8, strided<[1], offset: ?>>
    %initial = arith.cmpi eq, %t, %z : index
    scf.if %initial {
      mxint8gemv_ball.gemv_mxint8 <a = 1, b = 2, c = 3, k = 32, a_codes = 2, a_exponents = 40, b_base = 3, c_base = 4, accumulate = false, finalize = false>
      %partial = memref.subview %out[0] [272] [1] : memref<336xi8> to memref<272xi8, strided<[1]>>
      bbacc.mvout_strided %partial, 3 at %four rows %seventeen memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<272xi8, strided<[1]>>
    } else {
      mxint8gemv_ball.gemv_mxint8 <a = 1, b = 2, c = 3, k = 32, a_codes = 2, a_exponents = 40, b_base = 3, c_base = 4, accumulate = true, finalize = true>
      %result = memref.subview %out[272] [48] [1] : memref<336xi8> to memref<48xi8, strided<[1], offset: 272>>
      bbacc.mvout_strided %result, 3 at %four rows %three memory_stride %sixteen bytes %sixteen block_log2 %z <group = 0, bank_stride = 1> : memref<48xi8, strided<[1], offset: 272>>
    }
  }
  bbacc.mset <bank = 1, alloc = false, row = 0, col = 0>
  bbacc.mset <bank = 2, alloc = false, row = 0, col = 0>
  bbacc.mset <bank = 3, alloc = false, row = 0, col = 0>
  bbacc.mset <bank = 15, alloc = true, row = 1, col = 1>
  bbacc.mvin %out <bank = 15, depth = 21, stride = 1> : memref<336xi8>
  bbacc.mset <bank = 15, alloc = false, row = 0, col = 0>
  return
}
