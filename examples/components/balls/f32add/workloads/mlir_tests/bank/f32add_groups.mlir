// The reduce core has four banks: groups 0 and 1 share a two-group source
// bank, group 2 then reuses its banks. The first add_f32 starts from +0
// instead of the poison in the accumulator.
func.func @reduce_groups(%pair: memref<2x2x4xi32>, %last: memref<2x4xi32>,
                         %poison: memref<2x4xi32>, %output: memref<2x4xi32>)
    attributes {llvm.emit_c_interface} {
  %source = buckyball.bank_alloc <{row = 1 : i64, col = 2 : i64}>
  %accumulator = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %sum = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %groups = buckyball.bank_mvin %pair, %source <{depth = 2 : i64, stride = 1 : i64}> : memref<2x2x4xi32>
  %poisoned = buckyball.bank_mvin %poison, %accumulator <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  %first = buckyball.bank_custom "add_f32" (%groups, %poisoned) -> %sum {rows = 2 : i64, group = 0 : i64, first = true}
  %second = buckyball.bank_custom "add_f32" (%groups, %first) -> %poisoned {rows = 2 : i64, group = 1 : i64, first = false}
  buckyball.bank_release %groups
  %single = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %group2 = buckyball.bank_mvin %last, %single <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  %third = buckyball.bank_custom "add_f32" (%group2, %second) -> %first {rows = 2 : i64, group = 0 : i64, first = false}
  %stored = buckyball.bank_mvout %output, %third <{depth = 2 : i64, stride = 1 : i64}> : memref<2x4xi32>
  buckyball.bank_release %group2
  buckyball.bank_release %second
  buckyball.bank_release %stored
  return
}
