// CHECK: buckyball.relu
// CHECK-NOT: buckyball.bank_relu

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %group = arith.constant 0 : i64
  %lines = arith.constant 1024 : i64
  %bank = buckyball.bank_alloc {col = 1 : i64}
  %result = buckyball.bank_relu %bank %group %lines %lines : i64 i64 i64 i64
  buckyball.bank_release %result : i64
  return %zero : i8
}
