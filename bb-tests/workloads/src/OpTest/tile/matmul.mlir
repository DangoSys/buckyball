// Tile Dialect matmul test
// Matrix: 1024x1024 (fp32) @ 1024x1024 (fp32) -> 1024x1024 (fp32)
// Tests tile pass's auto-tiling capability (K-dimension splitting required)
// Verification done in C wrapper to avoid fp constant pool (.LCPI)

// C function that checks c[0][0] bit pattern == 0x44800000 (1024.0f)
func.func private @check_result(memref<1024x1024xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() : memref<1024x1024xf32>
  %b = memref.alloc() : memref<1024x1024xf32>
  %c = memref.alloc() : memref<1024x1024xf32>

  // Initialize using linalg.fill (compact lowering)
  linalg.fill ins(%one_f32 : f32) outs(%a : memref<1024x1024xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<1024x1024xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<1024x1024xf32>)

  // Tile matmul: compiler must split K=1024 into multiple K=16 mul_warp16 calls
  tile.tile_matmul %a %b %c
    : memref<1024x1024xf32> memref<1024x1024xf32> memref<1024x1024xf32>

  // Hand off to C for verification (avoids fp constant pool in MLIR)
  func.call @check_result(%c) : (memref<1024x1024xf32>) -> ()

  memref.dealloc %a : memref<1024x1024xf32>
  memref.dealloc %b : memref<1024x1024xf32>
  memref.dealloc %c : memref<1024x1024xf32>

  return %zero_i8 : i8
}
