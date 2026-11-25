// RUN: buddy-opt %s -lower-acc -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @reduce_groups
// CHECK-DAG: %[[F76:.*]] = llvm.mlir.constant(76 : i32) : i32
// CHECK-DAG: %[[AB:[^ ]+]] = arith.constant 2149581824 : i64
// CHECK-DAG: %[[DC:[^ ]+]] = arith.constant 2148534275 : i64
// CHECK-DAG: %[[ZERO:[^ ]+]] = arith.constant 0 : i64
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[AB]], %[[ZERO]], %[[F76]])
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[DC]], %[[ZERO]], %[[F76]])
// The banks 0 (A), 1 (B) and 3 (D) are loaded by the caller: 2 = A + B,
// then 1 = D + 2.
func.func @reduce_groups() -> i64 attributes {llvm.emit_c_interface} {
  %output = arith.constant 1 : i64
  f32add_ball.add_f32 <{a = 0 : i64, b = 1 : i64, c = 2 : i64, rows = 2 : i64}>
  f32add_ball.add_f32 <{a = 3 : i64, b = 2 : i64, c = 1 : i64, rows = 2 : i64}>
  return %output : i64
}
