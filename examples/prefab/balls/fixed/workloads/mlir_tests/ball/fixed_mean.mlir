// ISA-DAG: arith.constant 390 : i64
// ISA-DAG: arith.constant 391 : i64
// ISA: llvm.call_intrinsic "llvm.riscv.buckyball.custom"
memref.global "private" constant @input : memref<8x4xi32> = dense<[[2147483647, -2147483648, 0, 1], [0, -1, 2, 3], [-2, -3, 100, -100], [10, 20, 30, 40], [2147483647, -2147483648, 1, 2], [-1, -2, 3, 4], [-3, -4, 101, -101], [12, 22, 32, 42]]>
memref.global "private" constant @zero : memref<8x2xi64> = dense<0>
memref.global "private" constant @divisors : memref<4x4xi32> = dense<2>
func.func private @check_result(memref<4x4xi32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %input = memref.get_global @input : memref<8x4xi32>
  %zero = memref.get_global @zero : memref<8x2xi64>
  %divisors = memref.get_global @divisors : memref<4x4xi32>
  %output = memref.alloc() alignment = 64 : memref<4x4xi32>
  bbacc.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mvin %input <{bank = 0 : i64, depth = 8 : i64, stride = 1 : i64}> : memref<8x4xi32>
  bbacc.mvin %zero <{bank = 1 : i64, depth = 8 : i64, stride = 1 : i64}> : memref<8x2xi64>
  fixed_ball.fixed <{a = 0 : i64, b = 1 : i64, c = 2 : i64, rows = 8 : i64, mode = 6 : i64, fraction = 24 : i64}>
  bbacc.mvin %divisors <{bank = 1 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  fixed_ball.fixed <{a = 2 : i64, b = 1 : i64, c = 0 : i64, rows = 4 : i64, mode = 7 : i64, fraction = 24 : i64}>
  bbacc.mvout %output <{bank = 0 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  bbacc.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  bbacc.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  bbacc.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<4x4xi32>) -> ()
  memref.dealloc %output : memref<4x4xi32>
  return %success : i8
}
