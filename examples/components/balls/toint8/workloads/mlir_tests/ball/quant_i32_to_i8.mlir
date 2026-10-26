func.func private @check_result(memref<4x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %twenty = arith.constant 20 : index
  %eight = arith.constant 8 : index
  %input = memref.alloc() alignment = 64 : memref<16x4xi32>
  %scales = memref.alloc() alignment = 64 : memref<4x4xf32>
  %output = memref.alloc() alignment = 64 : memref<4x16xi8>
  linalg.fill ins(%one_f32 : f32) outs(%scales : memref<4x4xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%output : memref<4x16xi8>)
  scf.for %position = %zero to %four step %one {
    scf.for %channel = %zero to %sixteen step %one {
      %position_base = arith.muli %position, %twenty : index
      %with_channel = arith.addi %position_base, %channel : index
      %value = arith.subi %with_channel, %eight : index
      %value_i32 = arith.index_cast %value : index to i32
      %channel_row = arith.divui %channel, %four : index
      %row_base = arith.muli %position, %four : index
      %row = arith.addi %row_base, %channel_row : index
      %lane = arith.remui %channel, %four : index
      memref.store %value_i32, %input[%row, %lane] : memref<16x4xi32>
    }
  }
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  buckyball.mvin %scales <{bank = 2 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  buckyball.custom "int32_to_int8" <{sources = array<i64: 1, 2>, destination = 3 : i64}> {rows = 4 : i64, relu = true}
  buckyball.mvout %output <{bank = 3 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  memref.dealloc %input : memref<16x4xi32>
  memref.dealloc %scales : memref<4x4xf32>
  memref.dealloc %output : memref<4x16xi8>
  return %zero_i8 : i8
}
