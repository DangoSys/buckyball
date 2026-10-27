// A three-term sum as a chain of add_f32 into fresh banks: the first sum
// lands in a bank holding poison, which it overwrites, and the second reuses
// a released source bank.
func.func @reduce_groups(%first: memref<2x4xi32>, %second: memref<2x4xi32>,
                         %last: memref<2x4xi32>, %poison: memref<2x4xi32>,
                         %output: memref<2x4xi32>)
    attributes {llvm.emit_c_interface} {
  %a = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %sum = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = buckyball.bank_mvin %first, %a <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  %lb = buckyball.bank_mvin %second, %b <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  %poisoned = buckyball.bank_mvin %poison, %sum <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  %partial = f32add_ball.bank_add_f32 %la, %lb -> %poisoned <{rows = 2 : i64}>
  buckyball.bank_release %la
  %d = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %ld = buckyball.bank_mvin %last, %d <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  %total = f32add_ball.bank_add_f32 %ld, %partial -> %lb <{rows = 2 : i64}>
  %stored = buckyball.bank_mvout %output, %total <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  buckyball.bank_release %ld
  buckyball.bank_release %partial
  buckyball.bank_release %stored
  return
}
