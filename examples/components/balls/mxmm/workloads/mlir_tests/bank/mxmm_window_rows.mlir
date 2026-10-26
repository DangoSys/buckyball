// RUN: buddy-opt %s -assign-physical-banks | FileCheck %s --check-prefix=BANK
// BANK: buckyball.mxfp8_window
// RUN: buddy-opt %s -assign-physical-banks -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @window
// CHECK-DAG: %[[F75:.*]] = llvm.mlir.constant(75 : i32) : i32
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"({{.*}}, %[[F75]])
func.func @window(%a: i64, %b: i64, %c: i64, %rows: i64, %cols: i64, %count: i64, %full: i64, %start: i64, %first: i1, %last: i1, %base: i64) attributes {llvm.emit_c_interface} {
  %state = buckyball.bank_mxfp8_window %a %b %c %rows %cols %count %full %start %first %last %base : i64
  return
}
