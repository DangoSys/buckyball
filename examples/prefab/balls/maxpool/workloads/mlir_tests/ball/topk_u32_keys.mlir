memref.global "private" constant @keys : memref<3x4xi32> = dense<[[0, -1, 17, 17], [-2147483648, 0, 7, -1], [-2147483648, 3, 17, -1]]>
func.func private @check_result(memref<3x4xi32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %input = memref.get_global @keys : memref<3x4xi32>
  %output = memref.alloc() : memref<3x4xi32>
  bbacc.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  bbacc.mvin %input <{bank = 1 : i64, depth = 3 : i64, stride = 1 : i64}> : memref<3x4xi32>
  maxpool_ball.topk_u32 <{input = 1 : i64, output = 2 : i64, rows = 3 : i64, count = 11 : i64, k = 5 : i64}>
  bbacc.mvout %output <{bank = 2 : i64, depth = 3 : i64, stride = 1 : i64}> : memref<3x4xi32>
  bbacc.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  bbacc.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<3x4xi32>) -> ()
  memref.dealloc %output : memref<3x4xi32>
  return %success : i8
}
