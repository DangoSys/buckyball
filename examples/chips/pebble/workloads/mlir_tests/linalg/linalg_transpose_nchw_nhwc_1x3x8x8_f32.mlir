func.func private @check_result(memref<1x8x8x3xf32>) -> ()

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %in = memref.alloc() : memref<1x3x8x8xf32>
  %out = memref.alloc() : memref<1x8x8x3xf32>
  scf.for %c = %c0 to %c3 step %c1 {
    scf.for %h = %c0 to %c8 step %c1 {
      scf.for %w = %c0 to %c8 step %c1 {
        %cf = arith.index_cast %c : index to i32
        %hf = arith.index_cast %h : index to i32
        %wf = arith.index_cast %w : index to i32
        %s = arith.addi %cf, %hf : i32
        %s2 = arith.addi %s, %wf : i32
        %v = arith.sitofp %s2 : i32 to f32
        memref.store %v, %in[%c0, %c, %h, %w] : memref<1x3x8x8xf32>
      }
    }
  }
  linalg.transpose ins(%in : memref<1x3x8x8xf32>)
      outs(%out : memref<1x8x8x3xf32>) permutation = [0, 2, 3, 1]
  func.call @check_result(%out) : (memref<1x8x8x3xf32>) -> ()
  memref.dealloc %in : memref<1x3x8x8xf32>
  memref.dealloc %out : memref<1x8x8x3xf32>
  return %z : i8
}
