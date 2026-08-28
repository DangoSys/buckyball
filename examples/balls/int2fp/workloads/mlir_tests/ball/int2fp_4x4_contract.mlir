// CHECK: buckyball.intr.custom
// CHECK-NOT: buckyball.int2fp_tensor

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 4 : i64
  %da_addr = arith.constant 0 : i64
  %dw_addr = arith.constant 16 : i64
  %in0 = buckyball.bank_alloc
  %out0 = buckyball.bank_alloc
  buckyball.int2fp_tensor %in0, %out0, %iter, %da_addr, %dw_addr : i64
  buckyball.bank_release %in0 : i64
  buckyball.bank_release %out0 : i64
  return %z : i8
}
