memref.global "private" constant @input : memref<2x4xi32> = dense<[[-2147483648, -512, -511, -1], [0, 1, 511, 2147483647]]>
func.func private @check_result(memref<2x4xi32>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %rows = arith.constant 256 : index
  %three_i32 = arith.constant 3 : i32
  %hundred = arith.constant 100 : i32
  %input = memref.get_global @input : memref<2x4xi32>
  %table = memref.alloc() : memref<256x4xi32>
  %output = memref.alloc() : memref<2x4xi32>
  scf.for %row = %zero to %rows step %one {
    scf.for %lane = %zero to %four step %one {
      %base = arith.muli %row, %four : index
      %index = arith.addi %base, %lane : index
      %i = arith.index_cast %index : index to i32
      %scaled = arith.muli %i, %three_i32 : i32
      %value = arith.subi %scaled, %hundred : i32
      memref.store %value, %table[%row, %lane] : memref<256x4xi32>
    }
  }
  %a = bbacc.bank_alloc <row = 1, col = 1>
  %b = bbacc.bank_alloc <row = 1, col = 1>
  %c = bbacc.bank_alloc <row = 1, col = 1>
  %x = bbacc.bank_mvin %input, %a <depth = 2, stride = 1> : memref<2x4xi32>
  %t = bbacc.bank_mvin %table, %b <depth = 256, stride = 1> : memref<256x4xi32>
  %y = lut_ball.bank_lookup_i32 %x, %t -> %c <rows = 2>
  %stored = bbacc.bank_mvout %output, %y <depth = 2, stride = 1> : memref<2x4xi32>
  bbacc.bank_release %x
  bbacc.bank_release %t
  bbacc.bank_release %stored
  func.call @check_result(%output) : (memref<2x4xi32>) -> ()
  memref.dealloc %table : memref<256x4xi32>
  memref.dealloc %output : memref<2x4xi32>
  %success = arith.constant 0 : i8
  return %success : i8
}
