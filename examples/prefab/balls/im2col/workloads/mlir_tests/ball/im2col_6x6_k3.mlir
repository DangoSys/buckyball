// Physical-bank im2col_i8: one group of sixteen channels, 6x6, k3, 16 windows
// in M tiles of eight rows.

func.func private @check_result(memref<144x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c16 = arith.constant 16 : index

  %input = memref.alloc() alignment = 64 : memref<36x16xi8>
  %output = memref.alloc() alignment = 64 : memref<144x16xi8>
  linalg.fill ins(%zero_i8 : i8) outs(%output : memref<144x16xi8>)

  scf.for %r = %c0 to %c6 step %c1 {
    scf.for %col = %c0 to %c6 step %c1 {
      scf.for %ch = %c0 to %c16 step %c1 {
        %tmp = arith.muli %r, %c6 : index
        %pixel = arith.addi %tmp, %col : index
        %sum = arith.addi %r, %col : index
        %value = arith.addi %sum, %ch : index
        %v8 = arith.index_cast %value : index to i8
        memref.store %v8, %input[%pixel, %ch] : memref<36x16xi8>
      }
    }
  }

  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 36 : i64, stride = 1 : i64}> : memref<36x16xi8>
  im2col_ball.im2col_i8 <{input = 1 : i64, output = 2 : i64, input_side = 6 : i64, kernel = 3 : i64, stride = 1 : i64, padding = 0 : i64, tile_rows = 8 : i64}>
  buckyball.mvout %output <{bank = 2 : i64, depth = 144 : i64, stride = 1 : i64}> : memref<144x16xi8>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>

  func.call @check_result(%output) : (memref<144x16xi8>) -> ()
  memref.dealloc %input : memref<36x16xi8>
  memref.dealloc %output : memref<144x16xi8>
  return %zero_i8 : i8
}
