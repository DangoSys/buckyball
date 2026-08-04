func.func private @check_wt(memref<256x120xf32>) -> ()
func.func private @check_result(memref<1x120xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %w = memref.alloc() : memref<120x256xf32>
  %wt = memref.alloc() : memref<256x120xf32>
  %a = memref.alloc() : memref<1x256xf32>
  %c = memref.alloc() : memref<1x120xf32>
  linalg.fill ins(%one : f32) outs(%w : memref<120x256xf32>)
  linalg.fill ins(%one : f32) outs(%a : memref<1x256xf32>)
  linalg.fill ins(%zero : f32) outs(%c : memref<1x120xf32>)
  tile.tile_transpose %w %wt : memref<120x256xf32> memref<256x120xf32>
  func.call @check_wt(%wt) : (memref<256x120xf32>) -> ()
  tile.tile_matmul %a %wt %c : memref<1x256xf32> memref<256x120xf32> memref<1x120xf32>
  func.call @check_result(%c) : (memref<1x120xf32>) -> ()
  memref.dealloc %w : memref<120x256xf32>
  memref.dealloc %wt : memref<256x120xf32>
  memref.dealloc %a : memref<1x256xf32>
  memref.dealloc %c : memref<1x120xf32>
  return %zero_i8 : i8
}
