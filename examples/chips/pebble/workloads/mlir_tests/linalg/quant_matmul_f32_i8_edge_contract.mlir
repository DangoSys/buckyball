// Edge tiles are padded in lowering: one FP2INT, eight SMATMUL and eight
// channel INT2FP instructions. Pebble owns these funct7 assignments.
// CHECK: funct7 = 51
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54
// CHECK: funct7 = 65
// CHECK: funct7 = 54

func.func @main(%a: memref<1x256xf32>, %b: memref<256x120xi8>,
                %c: memref<1x120xf32>) {
  tile.tile_matmul %a %b %c {dw_addr = 16 : i64, dw_bytes = 512 : i64,
                              per_channel = true}
      : memref<1x256xf32> memref<256x120xi8> memref<1x120xf32>
  return
}
