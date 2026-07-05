// Tile Dialect matmul test: 1x84 @ 84x10
// Mirrors the LeNet classifier tail and exercises tile-level padding.

func.func private @check_result(memref<1x10xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() : memref<1x84xf32>
  %b = memref.alloc() : memref<84x10xf32>
  %c = memref.alloc() : memref<1x10xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<1x84xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<84x10xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<1x10xf32>)

  tile.tile_matmul %a %b %c
    : memref<1x84xf32> memref<84x10xf32> memref<1x10xf32>

  func.call @check_result(%c) : (memref<1x10xf32>) -> ()

  memref.dealloc %a : memref<1x84xf32>
  memref.dealloc %b : memref<84x10xf32>
  memref.dealloc %c : memref<1x10xf32>

  return %zero_i8 : i8
}
