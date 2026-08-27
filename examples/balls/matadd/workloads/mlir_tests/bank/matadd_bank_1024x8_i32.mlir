func.func private @check_result(memref<1024x8xi32>) -> ()

func.func @main() -> i8 {
  %zeroI8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %lines = arith.constant 1024 : i64
  %stride = arith.constant 1 : i64
  %limit = arith.constant 1024 : index
  %columns = arith.constant 8 : index
  %a = memref.alloc() alignment = 64 : memref<1024x8xi32>
  %b = memref.alloc() alignment = 64 : memref<1024x8xi32>
  %c = memref.alloc() alignment = 64 : memref<1024x8xi32>

  scf.for %line = %zero to %limit step %one {
    scf.for %column = %zero to %columns step %one {
      %aValue = arith.index_cast %line : index to i32
      %bValue = arith.index_cast %column : index to i32
      memref.store %aValue, %a[%line, %column] : memref<1024x8xi32>
      memref.store %bValue, %b[%line, %column] : memref<1024x8xi32>
    }
  }

  %aBank = buckyball.bank_alloc {col = 2 : i64}
  %bBank = buckyball.bank_alloc {col = 2 : i64}
  %cBank = buckyball.bank_alloc {col = 2 : i64}
  %aLoaded = buckyball.bank_mvin %a %aBank %lines %stride
      : memref<1024x8xi32> i64 i64 i64
  %bLoaded = buckyball.bank_mvin %b %bBank %lines %stride
      : memref<1024x8xi32> i64 i64 i64
  %result = buckyball.bank_matadd %aLoaded %bLoaded %cBank %lines : i64 i64 i64 i64
  %stored = buckyball.bank_mvout %c %result %lines %stride
      : memref<1024x8xi32> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %aLoaded : i64
  buckyball.bank_release %bLoaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%c) : (memref<1024x8xi32>) -> ()
  memref.dealloc %a : memref<1024x8xi32>
  memref.dealloc %b : memref<1024x8xi32>
  memref.dealloc %c : memref<1024x8xi32>
  return %zeroI8 : i8
}
