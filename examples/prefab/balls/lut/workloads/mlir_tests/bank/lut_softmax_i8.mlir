memref.global "private" constant @input : memref<2x4xi32> = dense<[[7, 6, 5, 4], [7, 6, 5, 4]]>
func.func private @check_result(memref<2x16xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %z = arith.constant 0 : i32
  %v0 = arith.constant 1073741824 : i32
  %v1 = arith.constant 536870912 : i32
  %v2 = arith.constant 268435456 : i32
  %v3 = arith.constant 134217728 : i32
  %input = memref.get_global @input : memref<2x4xi32>
  %table = memref.alloc() : memref<256x4xi32>
  linalg.fill ins(%z : i32) outs(%table : memref<256x4xi32>)
  memref.store %v0, %table[%zero, %zero] : memref<256x4xi32>
  memref.store %v1, %table[%zero, %one] : memref<256x4xi32>
  memref.store %v2, %table[%zero, %two] : memref<256x4xi32>
  memref.store %v3, %table[%zero, %three] : memref<256x4xi32>
  %output = memref.alloc() : memref<2x16xi8>
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 1>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %x = bbacc.bank_mvin %input, %a <depth = 2, stride = 1> : memref<2x4xi32>
  %t = bbacc.bank_mvin %table, %b <depth = 256, stride = 1> : memref<256x4xi32>
  %y = lut_ball.bank_softmax_i8 %x, %t -> %c <rows = 2, multiplier = 7864320>
  %stored = bbacc.bank_mvout %output, %y <depth = 2, stride = 1> : memref<2x16xi8>
  bbacc.bank_release %x
  bbacc.bank_release %t
  bbacc.bank_release %stored
  func.call @check_result(%output) : (memref<2x16xi8>) -> ()
  memref.dealloc %table : memref<256x4xi32>
  memref.dealloc %output : memref<2x16xi8>
  %success = arith.constant 0 : i8
  return %success : i8
}
