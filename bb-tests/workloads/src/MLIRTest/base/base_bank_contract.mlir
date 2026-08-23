// CHECK: buckyball.intr.mset
// CHECK: buckyball.intr.mvin
// CHECK: buckyball.intr.mvout
// CHECK: buckyball.intr.mset
// CHECK-NOT: buckyball.bank_alloc
// CHECK-NOT: buckyball.bank_mvin
// CHECK-NOT: buckyball.bank_mvout
// CHECK-NOT: buckyball.bank_release

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %input = memref.alloc() : memref<16x16xi8>
  %output = memref.alloc() : memref<16x16xi8>

  %bank = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bank %depth %stride
    : memref<16x16xi8> i64 i64 i64
  %stored = buckyball.bank_mvout %output %loaded %depth %stride
    : memref<16x16xi8> i64 i64 i64
  buckyball.bank_release %stored : i64

  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
