// ISA: arith.constant 896 : i64
// ISA: llvm.call_intrinsic "llvm.riscv.buckyball.custom"
memref.global "private" constant @input : memref<4x4xi32> = dense<[[1, 3, -1, -3], [5, 7, -5, -7], [127, 128, -127, -128], [2147483647, -2147483648, 17, -17]]>
memref.global "private" constant @parameters : memref<8x4xi32> = dense<[[1,1,1,1],[1,1,1,1],[3,3,3,3],[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,1],[1,1,1,1]]>
func.func private @check_result(memref<4x4xi32>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.get_global @input : memref<4x4xi32>
  %parameters = memref.get_global @parameters : memref<8x4xi32>
  %output = memref.alloc() alignment = 64 : memref<4x4xi32>
  bbacc.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mvin %input <{bank = 0 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  bbacc.mvin %parameters <{bank = 1 : i64, depth = 8 : i64, stride = 1 : i64}> : memref<8x4xi32>
  fixed_ball.fixed <{a = 0 : i64, b = 1 : i64, c = 2 : i64, rows = 4 : i64, mode = 0 : i64, fraction = 24 : i64, stream}>
  bbacc.mvout %output <{bank = 2 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  bbacc.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  bbacc.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  bbacc.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<4x4xi32>) -> ()
  memref.dealloc %output : memref<4x4xi32>
  return %zero : i8
}
