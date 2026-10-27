func.func private @check_result(memref<4x16xi8>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  // 0x99: both nibbles 9, the value 1.
  %raw = arith.constant -103 : i8
  %scale = arith.constant 127 : i8
  %input = memref.alloc() : memref<2x16xi8>
  %scales = memref.alloc() : memref<1x16xi8>
  %output = memref.alloc() : memref<4x16xi8>
  linalg.fill ins(%raw : i8) outs(%input : memref<2x16xi8>)
  linalg.fill ins(%scale : i8) outs(%scales : memref<1x16xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%output : memref<4x16xi8>)
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 2 : i64, stride = 1 : i64}> : memref<2x16xi8>
  buckyball.mvin %scales <{bank = 2 : i64, depth = 1 : i64, stride = 1 : i64}> : memref<1x16xi8>
  mxfp2int_ball.mxfp4_i8 <{data = 1 : i64, scales = 2 : i64, output = 3 : i64, rows = 2 : i64}>
  buckyball.mvout %output <{bank = 3 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  memref.dealloc %input : memref<2x16xi8>
  memref.dealloc %scales : memref<1x16xi8>
  memref.dealloc %output : memref<4x16xi8>
  return %zero_i8 : i8
}
