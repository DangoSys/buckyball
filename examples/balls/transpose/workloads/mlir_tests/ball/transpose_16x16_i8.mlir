// Already-lowered ball Op: buckyball.transpose (physical-bank form).

func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %c16_i32 = arith.constant 16 : i32
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %iter = arith.constant 16 : i64
  %elem_bits = arith.constant 8 : i64

  %input = memref.alloc() alignment = 64 : memref<16x16xi8>
  %output = memref.alloc() alignment = 64 : memref<16x16xi8>

  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %mul = arith.muli %ii, %c16_i32 : i32
      %val = arith.addi %mul, %jj : i32
      %v8 = arith.trunci %val : i32 to i8
      memref.store %v8, %input[%i, %j] : memref<16x16xi8>
    }
  }

  %bin = buckyball.bank_alloc
  %bout = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %bin %depth %stride
      : memref<16x16xi8> i64 i64 i64
  buckyball.transpose %loaded, %bout, %iter, %elem_bits : i64
  %stored = buckyball.bank_mvout %output %bout %depth %stride
      : memref<16x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
