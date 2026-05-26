// Tile Dialect matmul test
// Matrix: 127x17 (fp32) @ 17x127 (fp32) -> 127x127 (fp32)
// Tests tile-level padding before lowering to buckyball.matmul.

func.func private @check_result(memref<127x127xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() : memref<127x17xf32>
  %b = memref.alloc() : memref<17x127xf32>
  %c = memref.alloc() : memref<127x127xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<127x17xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<17x127xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<127x127xf32>)

  tile.tile_matmul %a %b %c
    : memref<127x17xf32> memref<17x127xf32> memref<127x127xf32>

  func.call @check_result(%c) : (memref<127x127xf32>) -> ()

  memref.dealloc %a : memref<127x17xf32>
  memref.dealloc %b : memref<17x127xf32>
  memref.dealloc %c : memref<127x127xf32>

  return %zero_i8 : i8
}
