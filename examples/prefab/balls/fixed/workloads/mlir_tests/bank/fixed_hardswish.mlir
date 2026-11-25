// BANK: fixed_ball.fixed
// BANK-SAME: mode = 1
// BANK-SAME: fraction = 24
memref.global "private" constant @input : memref<4x4xi32> = dense<[[0, 8388608, 16777216, -8388608], [-16777216, -25165824, 33554432, -33554432], [50331648, -50331648, 67108864, -67108864], [2147483647, -2147483648, 50331647, -50331647]]>
memref.global "private" constant @parameters : memref<8x4xi32> = dense<[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]>
func.func private @check_result(memref<4x4xi32>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %input = memref.get_global @input : memref<4x4xi32>
  %parameters = memref.get_global @parameters : memref<8x4xi32>
  %output = memref.alloc() alignment = 64 : memref<4x4xi32>
  %a = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %input, %a <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  %lb = bbacc.bank_mvin %parameters, %b <{depth = 8 : i64, stride = 1 : i64}> : memref<8x4xi32>
  %result = fixed_ball.bank_fixed %la, %lb -> %c <{rows = 4 : i64, mode = 1 : i64, fraction = 24 : i64}>
  %stored = bbacc.bank_mvout %output, %result <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  bbacc.bank_release %la
  bbacc.bank_release %lb
  bbacc.bank_release %stored
  func.call @check_result(%output) : (memref<4x4xi32>) -> ()
  memref.dealloc %output : memref<4x4xi32>
  return %zero : i8
}
