func.func private @check_result(memref<1x100x84xf32>) -> ()

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c84 = arith.constant 84 : index
  %c100 = arith.constant 100 : index
  %in = memref.alloc() : memref<1x84x100xf32>
  %out = memref.alloc() : memref<1x100x84xf32>
  scf.for %a = %c0 to %c84 step %c1 {
    scf.for %b = %c0 to %c100 step %c1 {
      %ai = arith.index_cast %a : index to i32
      %bi = arith.index_cast %b : index to i32
      %s = arith.addi %ai, %bi : i32
      %v = arith.sitofp %s : i32 to f32
      memref.store %v, %in[%c0, %a, %b] : memref<1x84x100xf32>
    }
  }
  linalg.transpose ins(%in : memref<1x84x100xf32>)
      outs(%out : memref<1x100x84xf32>) permutation = [0, 2, 1]
  func.call @check_result(%out) : (memref<1x100x84xf32>) -> ()
  memref.dealloc %in : memref<1x84x100xf32>
  memref.dealloc %out : memref<1x100x84xf32>
  return %z : i8
}
