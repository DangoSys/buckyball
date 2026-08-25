// Already-lowered ball Op: buckyball.smatmul (physical-bank form).

func.func private @check_result(memref<16x16xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %cfg = arith.constant 268501008 : i64

  %a = memref.alloc() alignment = 64 : memref<16x16xi8>
  %b = memref.alloc() alignment = 64 : memref<16x16xi8>
  %c = memref.alloc() alignment = 64 : memref<16x16xi32>

  linalg.fill ins(%one_i8 : i8) outs(%a : memref<16x16xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%b : memref<16x16xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%c : memref<16x16xi32>)

  %ba = buckyball.bank_alloc
  %bb = buckyball.bank_alloc
  %bc = buckyball.bank_alloc {col = 4 : i64}
  %la = buckyball.bank_mvin %a %ba %depth %stride
      : memref<16x16xi8> i64 i64 i64
  %lb = buckyball.bank_mvin %b %bb %depth %stride
      : memref<16x16xi8> i64 i64 i64
  buckyball.smatmul %la, %lb, %bc, %cfg : i64
  %stored = buckyball.bank_mvout %c %bc %depth %stride
      : memref<16x16xi32> i64 i64 i64
  buckyball.bank_release %la : i64
  buckyball.bank_release %lb : i64
  buckyball.bank_release %stored : i64

  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero_i8 : i8
}
