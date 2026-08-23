// One activation chunk feeds two N panels with a single tensor Dw at byte 16.
// Pebble owns these funct7 assignments.
// CHECK: funct7 = 51
// CHECK: funct7 = 65
// CHECK: funct7 = 52
// CHECK: funct7 = 65
// CHECK: funct7 = 52

func.func @main(%a: memref<16x32xf32>, %b: memref<32x32xi8>,
                %c: memref<16x32xf32>) {
  tile.tile_matmul %a %b %c {dw_addr = 16 : i64, dw_bytes = 4 : i64,
                              per_channel = false}
      : memref<16x32xf32> memref<32x32xi8> memref<16x32xf32>
  return
}
