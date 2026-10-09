func.func private @check_result(memref<16x48xi8>, memref<16x32xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c48 = arith.constant 48 : index
  %n48 = arith.constant 48 : i32
  %n32 = arith.constant 32 : i32
  %n91 = arith.constant 91 : i8
  %n7 = arith.constant 7 : i32
  %bank = arith.constant 9 : i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %stride2 = arith.constant 2 : i64
  %input = memref.alloc() alignment = 64 : memref<16x48xi8>
  %patch = memref.alloc() alignment = 64 : memref<16x32xi8>
  %output = memref.alloc() alignment = 64 : memref<16x48xi8>
  %selected = memref.alloc() alignment = 64 : memref<16x32xi8>
  scf.for %i = %c0 to %c16 step %c1 {
    %ii = arith.index_cast %i : index to i32
    scf.for %j = %c0 to %c48 step %c1 {
      %jj = arith.index_cast %j : index to i32
      %base = arith.muli %ii, %n48 : i32
      %value = arith.addi %base, %jj : i32
      %byte = arith.trunci %value : i32 to i8
      memref.store %byte, %input[%i, %j] : memref<16x48xi8>
    }
    scf.for %j = %c0 to %c32 step %c1 {
      %jj = arith.index_cast %j : index to i32
      %base = arith.muli %ii, %n32 : i32
      %value = arith.addi %base, %jj : i32
      %shifted = arith.addi %value, %n7 : i32
      %byte = arith.trunci %shifted : i32 to i8
      memref.store %byte, %patch[%i, %j] : memref<16x32xi8>
      memref.store %n91, %selected[%i, %j] : memref<16x32xi8>
    }
  }
  buckyball.mset %bank <row = 1, col = 3> : i64
  buckyball.mvin %input %bank %depth %stride : memref<16x48xi8> i64 i64 i64
  buckyball.mvin %patch %bank %depth %stride2 <group = 1> : memref<16x32xi8> i64 i64 i64
  buckyball.mvout %selected %bank %depth %stride2 <group = 1> : memref<16x32xi8> i64 i64 i64
  buckyball.mvout %output %bank %depth %stride : memref<16x48xi8> i64 i64 i64
  buckyball.fence
  buckyball.mset %bank <alloc = false, row = 0, col = 0> : i64
  func.call @check_result(%output, %selected) : (memref<16x48xi8>, memref<16x32xi8>) -> ()
  memref.dealloc %input : memref<16x48xi8>
  memref.dealloc %patch : memref<16x32xi8>
  memref.dealloc %output : memref<16x48xi8>
  memref.dealloc %selected : memref<16x32xi8>
  return %zero : i8
}
