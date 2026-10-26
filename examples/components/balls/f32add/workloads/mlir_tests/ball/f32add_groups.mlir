// RUN: buddy-opt %s -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @reduce_groups
// CHECK-DAG: %[[F76:.*]] = llvm.mlir.constant(76 : i32) : i32
// CHECK-DAG: %[[AB:[^ ]+]] = arith.constant 2183169056 : i64
// CHECK-DAG: %[[DC:[^ ]+]] = arith.constant 2182121507 : i64
// CHECK-DAG: %[[ZERO:[^ ]+]] = arith.constant 0 : i64
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[AB]], %[[ZERO]], %[[F76]])
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[DC]], %[[ZERO]], %[[F76]])
// The banks 32 (A), 33 (B) and 35 (D) are loaded by the caller: 34 = A + B,
// then 33 = D + 34.
func.func @reduce_groups() -> i64 attributes {llvm.emit_c_interface} {
  %output = arith.constant 33 : i64
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 2 : i64}
  buckyball.custom "add_f32" <{sources = array<i64: 35, 34>, destination = 33 : i64}> {rows = 2 : i64}
  return %output : i64
}
