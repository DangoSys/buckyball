func.func private @check_result(memref<64x4xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %eight = arith.constant 8 : index
  %sixteen = arith.constant 16 : index

  %input0 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %input1 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %weight0 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %weight1 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %bias = memref.alloc() alignment = 64 : memref<4x4xi32>
  %output = memref.alloc() alignment = 64 : memref<64x4xi32>
  linalg.fill ins(%zero_i32 : i32) outs(%output : memref<64x4xi32>)

  scf.for %row = %zero to %sixteen step %one {
    scf.for %column = %zero to %sixteen step %one {
      %row_i32 = arith.index_cast %row : index to i32
      %column_i32 = arith.index_cast %column : index to i32
      %sum = arith.addi %row_i32, %column_i32 : i32
      %difference = arith.subi %row_i32, %column_i32 : i32
      %twice_row = arith.addi %row_i32, %row_i32 : i32
      %second = arith.subi %twice_row, %column_i32 : i32
      %sum_i8 = arith.trunci %sum : i32 to i8
      %difference_i8 = arith.trunci %difference : i32 to i8
      %second_i8 = arith.trunci %second : i32 to i8
      memref.store %difference_i8, %input0[%row, %column] : memref<16x16xi8>
      memref.store %sum_i8, %weight0[%row, %column] : memref<16x16xi8>
      memref.store %sum_i8, %input1[%row, %column] : memref<16x16xi8>
      memref.store %second_i8, %weight1[%row, %column] : memref<16x16xi8>
    }
    %bias_value = arith.subi %row, %eight : index
    %bias_i32 = arith.index_cast %bias_value : index to i32
    %bias_row = arith.divui %row, %four : index
    %bias_lane = arith.remui %row, %four : index
    memref.store %bias_i32, %bias[%bias_row, %bias_lane] : memref<4x4xi32>
  }

  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 4 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input0 <{bank = 0 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mvin %weight0 <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mvin %input1 <{bank = 2 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mvin %weight1 <{bank = 3 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  // The bias occupies the first four rows of C.
  buckyball.mvin %bias <{bank = 4 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  buckyball.custom "matmul_i8_block" <{sources = array<i64: 0, 1>, destination = 4 : i64, read_destination}> {rows = 16 : i64, k = 16 : i64, first = true, last = false}
  buckyball.custom "matmul_i8_block" <{sources = array<i64: 2, 3>, destination = 4 : i64, read_destination}> {rows = 16 : i64, k = 16 : i64, first = false, last = true}
  buckyball.mvout %output <{bank = 4 : i64, depth = 64 : i64, stride = 1 : i64}> : memref<64x4xi32>
  func.call @check_result(%output) : (memref<64x4xi32>) -> ()
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 4 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  memref.dealloc %input0 : memref<16x16xi8>
  memref.dealloc %input1 : memref<16x16xi8>
  memref.dealloc %weight0 : memref<16x16xi8>
  memref.dealloc %weight1 : memref<16x16xi8>
  memref.dealloc %bias : memref<4x4xi32>
  memref.dealloc %output : memref<64x4xi32>
  return %zero_i8 : i8
}
