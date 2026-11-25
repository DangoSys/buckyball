memref.global "private" constant @keys : memref<3x8xi32> = dense<[[5, 0, 5, -1, -2147483648, 6, 6, 0], [-1, -2147483648, 1, 2, 3, 4, 5, 6], [7, 8, 9, -1, -1, -1, -1, -1]]>
// BANK: maxpool_ball.topk_u32
func.func private @check_result(memref<4x4xi32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %input = memref.get_global @keys : memref<3x8xi32>
  %output = memref.alloc() : memref<4x4xi32>
  %a = bbacc.bank_alloc <{row = 1 : i64, col = 2 : i64}>
  %c = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %loaded = bbacc.bank_mvin %input, %a <{depth = 3 : i64, stride = 1 : i64}> : memref<3x8xi32>
  %result = maxpool_ball.bank_topk_u32 %loaded -> %c <{rows = 3 : i64, count = 19 : i64, k = 7 : i64}>
  %stored = bbacc.bank_mvout %output, %result <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  bbacc.bank_release %loaded
  bbacc.bank_release %stored
  func.call @check_result(%output) : (memref<4x4xi32>) -> ()
  memref.dealloc %output : memref<4x4xi32>
  return %success : i8
}
