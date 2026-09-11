func.func private @check_result(memref<4x16xi8>) -> ()
func.func @main() -> i8 {
  %z8 = arith.constant 0 : i8
  %raw = arith.constant -103 : i8
  %scale = arith.constant 127 : i8
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : i64
  %one64 = arith.constant 1 : i64
  %four = arith.constant 4 : i64
  %sixteen = arith.constant 16 : i64
  %zero64 = arith.constant 0 : i64
  %stride = arith.constant 1 : i64
  %input = memref.alloc() : memref<2x16xi8>
  %scales = memref.alloc() : memref<1x16xi8>
  %output = memref.alloc() : memref<4x16xi8>
  linalg.fill ins(%raw : i8) outs(%input : memref<2x16xi8>)
  linalg.fill ins(%scale : i8) outs(%scales : memref<1x16xi8>)
  linalg.fill ins(%z8 : i8) outs(%output : memref<4x16xi8>)
  %in = arith.constant 0 : i64
  %out = arith.constant 1 : i64
  buckyball.mset %in {row = 1 : i64, col = 1 : i64} : i64
  buckyball.mset %out {row = 1 : i64, col = 1 : i64} : i64
  buckyball.mvin %input %in %two %stride : memref<2x16xi8> i64 i64 i64
  buckyball.mvin_mmio %scales %zero64 %one64 %sixteen : memref<1x16xi8> i64 i64 i64
  buckyball.mxfp2int %in, %out, %two, %zero64 : i64
  buckyball.mvout %output %out %four %stride : memref<4x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  buckyball.mset %in {alloc = false, row = 0 : i64, col = 0 : i64} : i64
  buckyball.mset %out {alloc = false, row = 0 : i64, col = 0 : i64} : i64
  memref.dealloc %input : memref<2x16xi8>
  memref.dealloc %scales : memref<1x16xi8>
  memref.dealloc %output : memref<4x16xi8>
  return %z8 : i8
}
