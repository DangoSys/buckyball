// CHECK: buckyball.intr.mset
// CHECK: buckyball.intr.mvin
// CHECK: buckyball.intr.mvout
// CHECK: buckyball.intr.fence
// CHECK-NOT: buckyball.mset
// CHECK-NOT: buckyball.mvin
// CHECK-NOT: buckyball.mvout
// CHECK-NOT: buckyball.fence

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %bank = arith.constant 0 : i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %input = memref.alloc() : memref<16x16xi8>
  %output = memref.alloc() : memref<16x16xi8>

  buckyball.mset %bank {row = 1 : i64, col = 1 : i64} : i64
  buckyball.mvin %input %bank %depth %stride
    : memref<16x16xi8> i64 i64 i64
  buckyball.mvout %output %bank %depth %stride
    : memref<16x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.mset %bank {alloc = false, row = 0 : i64, col = 0 : i64} : i64

  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
