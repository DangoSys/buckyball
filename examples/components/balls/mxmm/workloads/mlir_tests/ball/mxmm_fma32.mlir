func.func @step_m1_n16_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807361 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m1_n16_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807361 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m1_n16_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807361 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m1_n48_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938433 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m1_n48_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938433 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m1_n48_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938433 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m16_n16_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807376 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m16_n16_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807376 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m16_n16_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807376 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m16_n48_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938448 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m16_n48_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938448 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m16_n48_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938448 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m32_n16_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807392 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m32_n16_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807392 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m32_n16_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073807392 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m32_n48_first() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938464 : i64 %first = arith.constant true %last = arith.constant false %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m32_n48_last() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938464 : i64 %first = arith.constant false %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
func.func @step_m32_n48_single() attributes {llvm.emit_c_interface} {
  %a = arith.constant 3 : i64 %b = arith.constant 4 : i64 %c = arith.constant 5 : i64 %shape = arith.constant 1073938464 : i64 %first = arith.constant true %last = arith.constant true %base = arith.constant 1 : i64
  buckyball.matmul_f32 %a, %b, %c, %shape, %first, %last, %base <fused = true> : i64
  return
}
