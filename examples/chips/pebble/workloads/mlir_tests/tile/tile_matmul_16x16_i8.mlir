func.func private @check_result(memref<16x16xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  %a = memref.alloc() alignment = 64 : memref<16x16xi8>
  %b = memref.alloc() alignment = 64 : memref<16x16xi8>
  %c = memref.alloc() alignment = 64 : memref<16x16xi32>
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %i3 = arith.muli %i, %c3 : index
      %j5 = arith.muli %j, %c5 : index
      %as = arith.addi %i3, %j5 : index
      %ar = arith.remui %as, %c7 : index
      %av0 = arith.subi %ar, %c3 : index
      %av = arith.index_cast %av0 : index to i8
      %i2 = arith.muli %i, %c2 : index
      %j3 = arith.muli %j, %c3 : index
      %bs = arith.addi %i2, %j3 : index
      %br = arith.remui %bs, %c5 : index
      %bv0 = arith.subi %br, %c2 : index
      %bv = arith.index_cast %bv0 : index to i8
      memref.store %av, %a[%i, %j] : memref<16x16xi8>
      memref.store %bv, %b[%i, %j] : memref<16x16xi8>
    }
  }
  linalg.fill ins(%zero_i32 : i32) outs(%c : memref<16x16xi32>)
  tile.tile_matmul %a %b %c : memref<16x16xi8> memref<16x16xi8> memref<16x16xi32>
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero_i8 : i8
}
