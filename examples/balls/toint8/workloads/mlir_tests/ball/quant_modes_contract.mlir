// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 51 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 67 : i32
// CHECK-NOT: buckyball.quant
// CHECK-NOT: buckyball.bank_quant

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %fpIter = arith.constant 4 : i64
  %i32Iter = arith.constant 64 : i64
  %scale = arith.constant 5.000000e-01 : f32
  %fpIn = arith.constant 0 : i64
  %i32In = arith.constant 1 : i64
  %scaleBank = arith.constant 2 : i64
  %fpOut = arith.constant 3 : i64
  %i8Out = arith.constant 4 : i64
  buckyball.quant_f32_to_i8 %fpIn, %fpOut, %fpIter, %scale : i64
  buckyball.quant_i32_to_i8
      %i32In, %scaleBank, %i8Out, %i32Iter {outputBase = 3 : i64,
      outputHeight = 4 : i64, outputStride = 6 : i64,
      outputWidth = 4 : i64, relu = true} : i64
  return %z : i8
}
