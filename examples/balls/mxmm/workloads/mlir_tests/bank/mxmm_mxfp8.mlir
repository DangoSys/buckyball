func.func @step_m1_n16_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032833 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m1_n16_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032833 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m1_n16_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032833 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m1_n48_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163905 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m1_n48_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163905 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m1_n48_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163905 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m16_n16_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032848 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m16_n16_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032848 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m16_n16_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032848 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m16_n48_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163920 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m16_n48_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163920 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m16_n48_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163920 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m32_n16_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032864 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m32_n16_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032864 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m32_n16_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295032864 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m32_n48_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163936 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m32_n48_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163936 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
func.func @step_m32_n48_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 4295163936 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  %state = buckyball.bank_mxfp8 %a %b %c %shape %first %last %base : i64 i64 i64 i64
  return
}
