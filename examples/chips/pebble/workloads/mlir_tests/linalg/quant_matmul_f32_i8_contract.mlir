// One activation chunk is quantized once; each N panel does SMATMUL + INT2FP.
// Pebble owns these funct7 assignments.
// CHECK: funct7 = 51
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54

func.func @main(%a: memref<16x32xf32>, %b: memref<32x32xi8>,
                %c: memref<16x32xf32>) {
  tile.tile_matmul %a %b %c {dw_addr = 16 : i64, dw_bytes = 128 : i64,
                              per_channel = true}
      : memref<16x32xf32> memref<32x32xi8> memref<16x32xf32>
  return
}
