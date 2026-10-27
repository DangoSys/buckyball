func.func private @check_result(memref<16x4xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %a = memref.alloc() : memref<16x4xi32>
  %b = memref.alloc() : memref<16x4xi32>
  %out = memref.alloc() : memref<16x4xi32>
  scf.for %row = %zero to %sixteen step %one {
    scf.for %lane = %zero to %four step %one {
      %r = arith.index_cast %row : index to i32
      %l = arith.index_cast %lane : index to i32
      %x = arith.addi %r, %l : i32
      %y = arith.subi %l, %r : i32
      memref.store %x, %a[%row, %lane] : memref<16x4xi32>
      memref.store %y, %b[%row, %lane] : memref<16x4xi32>
    }
  }
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %a <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  buckyball.mvin %b <{bank = 2 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  matadd_ball.add_i32 <{a = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 16 : i64}>
  buckyball.mvout %out <{bank = 3 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%out) : (memref<16x4xi32>) -> ()
  memref.dealloc %a : memref<16x4xi32>
  memref.dealloc %b : memref<16x4xi32>
  memref.dealloc %out : memref<16x4xi32>
  return %zero_i8 : i8
}
