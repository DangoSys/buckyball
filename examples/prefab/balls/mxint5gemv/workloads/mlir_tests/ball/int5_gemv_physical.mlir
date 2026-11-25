// LOWER-DAG: llvm.mlir.constant([[FUNCT]] : i32) : i32
module {
  func.func @compute(%arg0: memref<528xi8>, %arg1: memref<11264xi8>, %arg2: memref<336xi8>) attributes {llvm.emit_c_interface} {
  %base = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c32 = arith.constant 32 : index
    %c40 = arith.constant 40 : index
    %c176 = arith.constant 176 : index
    bbacc.mset <bank = 1, alloc = true, row = 1, col = 1>
    bbacc.mset <bank = 2, alloc = true, row = 1, col = 4>
    bbacc.mset <bank = 3, alloc = true, row = 1, col = 1>
    %subview = memref.subview %arg0[0] [512] [1] : memref<528xi8> to memref<512xi8, strided<[1]>>
    %subview_0 = memref.subview %arg0[512] [16] [1] : memref<528xi8> to memref<16xi8, strided<[1], offset: 512>>
    bbacc.mvin_strided %subview, 1 at %c2 rows %c32 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 0, bank_stride = 1> : memref<512xi8, strided<[1]>>
    bbacc.mvin_strided %subview_0, 1 at %c40 rows %c1 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 0, bank_stride = 1> : memref<16xi8, strided<[1], offset: 512>>
    %subview_1 = memref.subview %arg1[0] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1]>>
    %subview_2 = memref.subview %arg1[2816] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1], offset: 2816>>
    %subview_3 = memref.subview %arg1[5632] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1], offset: 5632>>
    %subview_4 = memref.subview %arg1[8448] [2816] [1] : memref<11264xi8> to memref<2816xi8, strided<[1], offset: 8448>>
    bbacc.mvin_strided %subview_1, 2 at %c3 rows %c176 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 0, bank_stride = 1> : memref<2816xi8, strided<[1]>>
    bbacc.mvin_strided %subview_2, 2 at %c3 rows %c176 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 1, bank_stride = 1> : memref<2816xi8, strided<[1], offset: 2816>>
    bbacc.mvin_strided %subview_3, 2 at %c3 rows %c176 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 2, bank_stride = 1> : memref<2816xi8, strided<[1], offset: 5632>>
    bbacc.mvin_strided %subview_4, 2 at %c3 rows %c176 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 3, bank_stride = 1> : memref<2816xi8, strided<[1], offset: 8448>>
    mxint5gemv_ball.gemv_mxint5 %base <a = 1, b = 2, c = 3, k = 512, a_codes = 2, a_exponents = 40, b_base = 3, accumulate = false, finalize = false, update = false>
    %subview_5 = memref.subview %arg2[0] [272] [1] : memref<336xi8> to memref<272xi8, strided<[1]>>
    bbacc.mvout_strided %subview_5, 3 at %c4 rows %c17 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 0, bank_stride = 1> : memref<272xi8, strided<[1]>>
    mxint5gemv_ball.gemv_mxint5 %base <a = 1, b = 2, c = 3, k = 512, a_codes = 2, a_exponents = 40, b_base = 3, accumulate = true, finalize = true, update = false>
    %subview_6 = memref.subview %arg2[272] [48] [1] : memref<336xi8> to memref<48xi8, strided<[1], offset: 272>>
    bbacc.mvout_strided %subview_6, 3 at %c4 rows %c3 memory_stride %c16 bytes %c16 block_log2 %c0 <group = 0, bank_stride = 1> : memref<48xi8, strided<[1], offset: 272>>
    bbacc.mset <bank = 2, alloc = false, row = 0, col = 0>
    bbacc.mset <bank = 3, alloc = false, row = 0, col = 0>
    bbacc.mvin %arg2 <bank = 1, depth = 21, stride = 1> : memref<336xi8>
    bbacc.mset <bank = 1, alloc = false, row = 0, col = 0>
    return
  }
}
