// Bank-level Op: buckyball.bank_im2col on 6x6 k3 (exact M tile).
// in[r,c] = (r + c) as i8; each spatial position is row lane 7.

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

  %iter = arith.constant 6 : i64
  %ksize = arith.constant 3 : i64
  %stride = arith.constant 1 : i64
  %padding = arith.constant 0 : i64
  %din = arith.constant 64 : i64
  %dout = arith.constant 16 : i64
  %s = arith.constant 1 : i64

  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %in %din %s
      : memref<64x16xi8> i64 i64 i64
  %inputBase = arith.constant 5 : i64
  %lane = arith.constant 7 : i64
  %next = buckyball.bank_im2col %loaded %out %iter %ksize %stride %padding
      %inputBase %lane {startRow = 0 : i64,
       startCol = 0 : i64, windowStart = 0 : i64, windowCount = 16 : i64}
      : i64 i64 i64 i64 i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %next %dout %s
      : memref<16x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<64x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
