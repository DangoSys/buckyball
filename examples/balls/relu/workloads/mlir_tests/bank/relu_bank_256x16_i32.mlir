// Full physical-bank ReLU test. One group holds 1024 128-bit lines.

func.func private @check_result(memref<1024x4xi32>) -> ()

func.func @main() -> i8 {
  %zeroI8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %group = arith.constant 0 : i64
  %lines = arith.constant 1024 : i64
  %stride = arith.constant 1 : i64
  %limit = arith.constant 1024 : index
  %lanes = arith.constant 4 : index
  %four = arith.constant 4 : index
  %seventeen = arith.constant 17 : i32
  %bias = arith.constant 8 : i32

  %input = memref.alloc() alignment = 64 : memref<1024x4xi32>
  %output = memref.alloc() alignment = 64 : memref<1024x4xi32>

  scf.for %line = %zero to %limit step %one {
    scf.for %lane = %zero to %lanes step %one {
      %base = arith.muli %line, %four : index
      %index = arith.addi %base, %lane : index
      %value = arith.index_cast %index : index to i32
      %remainder = arith.remui %value, %seventeen : i32
      %signed = arith.subi %remainder, %bias : i32
      memref.store %signed, %input[%line, %lane] : memref<1024x4xi32>
    }
  }

  %bank = buckyball.bank_alloc {col = 1 : i64}
  %loaded = buckyball.bank_mvin %input %bank %lines %stride
      : memref<1024x4xi32> i64 i64 i64
  %result = buckyball.bank_relu %loaded %group %lines %lines : i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %result %lines %stride
      : memref<1024x4xi32> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<1024x4xi32>) -> ()
  memref.dealloc %input : memref<1024x4xi32>
  memref.dealloc %output : memref<1024x4xi32>
  return %zeroI8 : i8
}
