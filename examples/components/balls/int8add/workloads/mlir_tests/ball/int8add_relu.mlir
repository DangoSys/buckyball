func.func private @check_result(memref<4x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %ratio = arith.constant 1.0 : f32
  %a = memref.alloc() : memref<4x16xi8>
  %b = memref.alloc() : memref<4x16xi8>
  %out = memref.alloc() : memref<4x16xi8>
  scf.for %row = %zero to %four step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %r = arith.index_cast %row : index to i32
      %l = arith.index_cast %lane : index to i32
      %a32 = arith.subi %l, %r : i32
      %b32 = arith.subi %r, %l : i32
      %a8 = arith.trunci %a32 : i32 to i8
      %b8 = arith.trunci %b32 : i32 to i8
      memref.store %a8, %a[%row, %lane] : memref<4x16xi8>
      memref.store %b8, %b[%row, %lane] : memref<4x16xi8>
    }
  }
  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %a <{bank = 0 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.mvin %b <{bank = 1 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.custom "add_i8" args(%ratio, %ratio : f32, f32) <{sources = array<i64: 0, 1>, destination = 2 : i64}> {rows = 4 : i64, relu = true}
  buckyball.mvout %out <{bank = 2 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%out) : (memref<4x16xi8>) -> ()
  memref.dealloc %a : memref<4x16xi8>
  memref.dealloc %b : memref<4x16xi8>
  memref.dealloc %out : memref<4x16xi8>
  return %zero_i8 : i8
}
