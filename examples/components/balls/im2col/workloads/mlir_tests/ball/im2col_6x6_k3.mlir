// Physical-bank im2col_i8: 6x6 input at rows 5.. lane 7, k3, 16 windows.

func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index

  %input = memref.alloc() alignment = 64 : memref<64x16xi8>
  %output = memref.alloc() alignment = 64 : memref<16x16xi8>
  linalg.fill ins(%zero_i8 : i8) outs(%input : memref<64x16xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%output : memref<16x16xi8>)

  scf.for %r = %c0 to %c6 step %c1 {
    scf.for %col = %c0 to %c6 step %c1 {
      %tmp = arith.muli %r, %c6 : index
      %idx = arith.addi %tmp, %col : index
      %bank_row = arith.addi %c5, %idx : index
      %ri = arith.index_cast %r : index to i32
      %ci = arith.index_cast %col : index to i32
      %v32 = arith.addi %ri, %ci : i32
      %v8 = arith.trunci %v32 : i32 to i8
      memref.store %v8, %input[%bank_row, %c7] : memref<64x16xi8>
    }
  }

  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 64 : i64, stride = 1 : i64}> : memref<64x16xi8>
  buckyball.custom "im2col_i8" <{sources = array<i64: 1>, destination = 2 : i64}> {input_side = 6 : i64, kernel = 3 : i64, stride = 1 : i64, padding = 0 : i64, start_row = 0 : i64, start_col = 0 : i64, op1_base = 5 : i64, lane = 7 : i64, window_start = 0 : i64, window_count = 16 : i64}
  buckyball.mvout %output <{bank = 2 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<64x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
