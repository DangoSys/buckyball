func.func private @check_result(memref<4x3x3x8xf32>) -> ()

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %in = memref.alloc() : memref<4x8x3x3xf32>
  %out = memref.alloc() : memref<4x3x3x8xf32>
  scf.for %o = %c0 to %c4 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %h = %c0 to %c3 step %c1 {
        scf.for %w = %c0 to %c3 step %c1 {
          %oi = arith.index_cast %o : index to i32
          %ii = arith.index_cast %i : index to i32
          %hi = arith.index_cast %h : index to i32
          %wi = arith.index_cast %w : index to i32
          %s = arith.addi %oi, %ii : i32
          %s2 = arith.addi %s, %hi : i32
          %s3 = arith.addi %s2, %wi : i32
          %v = arith.sitofp %s3 : i32 to f32
          memref.store %v, %in[%o, %i, %h, %w] : memref<4x8x3x3xf32>
        }
      }
    }
  }
  linalg.transpose ins(%in : memref<4x8x3x3xf32>)
      outs(%out : memref<4x3x3x8xf32>) permutation = [0, 2, 3, 1]
  func.call @check_result(%out) : (memref<4x3x3x8xf32>) -> ()
  memref.dealloc %in : memref<4x8x3x3xf32>
  memref.dealloc %out : memref<4x3x3x8xf32>
  return %z : i8
}
