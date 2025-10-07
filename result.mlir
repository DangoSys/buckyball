module {
  llvm.func @transpose_tile(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg7, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg8, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg9, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg10, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg12, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg11, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg13, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = builtin.unrealized_conversion_cast %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x32xi8>
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg0, %9[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg1, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg2, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg3, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg5, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg4, %14[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %arg6, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = builtin.unrealized_conversion_cast %16 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x32xi8>
    %subview = memref.subview %17[0, 0] [32, 32] [1, 1] : memref<32x32xi8> to memref<32x32xi8, strided<[32, 1]>>
    %subview_0 = memref.subview %8[0, 0] [32, 32] [1, 1] : memref<32x32xi8> to memref<32x32xi8, strided<[32, 1]>>
    %intptr = memref.extract_aligned_pointer_as_index %subview : memref<32x32xi8, strided<[32, 1]>> -> index
    %18 = arith.index_cast %intptr : index to i64
    %c1133871890432_i64 = arith.constant 1133871890432 : i64
    "buckyball.intr.bb_mvin"(%18, %c1133871890432_i64) : (i64, i64) -> ()
    %c8388608_i64 = arith.constant 8388608 : i64
    %c262144_i64 = arith.constant 262144 : i64
    "buckyball.intr.bb_transpose"(%c8388608_i64, %c262144_i64) : (i64, i64) -> ()
    %c8650784_i64 = arith.constant 8650784 : i64
    %c262144_i64_1 = arith.constant 262144 : i64
    "buckyball.intr.bb_transpose"(%c8650784_i64, %c262144_i64_1) : (i64, i64) -> ()
    %c8912912_i64 = arith.constant 8912912 : i64
    %c262144_i64_2 = arith.constant 262144 : i64
    "buckyball.intr.bb_transpose"(%c8912912_i64, %c262144_i64_2) : (i64, i64) -> ()
    %c9175088_i64 = arith.constant 9175088 : i64
    %c262144_i64_3 = arith.constant 262144 : i64
    "buckyball.intr.bb_transpose"(%c9175088_i64, %c262144_i64_3) : (i64, i64) -> ()
    %intptr_4 = memref.extract_aligned_pointer_as_index %subview_0 : memref<32x32xi8, strided<[32, 1]>> -> index
    %19 = arith.index_cast %intptr_4 : index to i64
    %c524800_i64 = arith.constant 524800 : i64
    "buckyball.intr.bb_mvout"(%19, %c524800_i64) : (i64, i64) -> ()
    llvm.return
  }
}
