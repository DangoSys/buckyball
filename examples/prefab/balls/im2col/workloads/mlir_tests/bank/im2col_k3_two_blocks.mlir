// Virtual-bank im2col_i8: an 8x8 map of 32 channels (two groups), k3 with
// padding 1, 64 windows spread over three output groups of six K tiles.

func.func private @check_result(memref<384x48xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %input = memref.alloc() alignment = 64 : memref<64x32xi8>
  %output = memref.alloc() alignment = 64 : memref<384x48xi8>
  scf.for %y = %c0 to %c8 step %c1 {
    scf.for %x = %c0 to %c8 step %c1 {
      scf.for %ch = %c0 to %c32 step %c1 {
        %row = arith.muli %y, %c8 : index
        %pixel = arith.addi %row, %x : index
        %scaled = arith.muli %ch, %c3 : index
        %value = arith.addi %pixel, %scaled : index
        %v8 = arith.index_cast %value : index to i8
        memref.store %v8, %input[%pixel, %ch] : memref<64x32xi8>
      }
    }
  }
  %input_bank = buckyball.bank_alloc <{row = 1 : i64, col = 2 : i64}>
  %output_bank = buckyball.bank_alloc <{row = 1 : i64, col = 3 : i64}>
  %loaded = buckyball.bank_mvin %input, %input_bank <{depth = 64 : i64, stride = 1 : i64}> : memref<64x32xi8>
  %matrix = im2col_ball.bank_im2col_i8 %loaded -> %output_bank <{input_side = 8 : i64, kernel = 3 : i64, stride = 1 : i64, padding = 1 : i64, tile_rows = 8 : i64}>
  %stored = buckyball.bank_mvout %output, %matrix <{depth = 384 : i64, stride = 1 : i64}> : memref<384x48xi8>
  buckyball.bank_release %loaded
  buckyball.bank_release %stored
  func.call @check_result(%output) : (memref<384x48xi8>) -> ()
  memref.dealloc %input : memref<64x32xi8>
  memref.dealloc %output : memref<384x48xi8>
  return %zero_i8 : i8
}
