// CHECK: buckyball.quant_f32_to_i8
// CHECK: buckyball.quant_i32_to_i8
// CHECK-SAME: outputBase = 3
// CHECK-SAME: outputHeight = 4
// CHECK-SAME: outputStride = 6
// CHECK-SAME: outputWidth = 4
// CHECK-SAME: relu = true
// CHECK-NOT: buckyball.bank_quant

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %fpIter = arith.constant 4 : i64
  %i32Iter = arith.constant 64 : i64
  %scale = arith.constant 5.000000e-01 : f32
  %fpIn = buckyball.bank_alloc
  %i32In = buckyball.bank_alloc
  %scaleBank = buckyball.bank_alloc
  %fpOut = buckyball.bank_alloc
  %i8Out = buckyball.bank_alloc
  %fpResult = buckyball.bank_quant_f32_to_i8 %fpIn %fpOut %fpIter %scale
      : i64 i64 i64 f32
  %i8Result = buckyball.bank_quant_i32_to_i8
      %i32In %scaleBank %i8Out %i32Iter {outputBase = 3 : i64,
      outputHeight = 4 : i64, outputStride = 6 : i64,
      outputWidth = 4 : i64, relu = true}
      : i64 i64 i64 i64
  buckyball.bank_release %fpIn : i64
  buckyball.bank_release %i32In : i64
  buckyball.bank_release %scaleBank : i64
  buckyball.bank_release %fpResult : i64
  buckyball.bank_release %i8Result : i64
  return %z : i8
}
