// RUN: buddy-opt %s -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @reduce_groups
// CHECK-DAG: %[[F76:.*]] = llvm.mlir.constant(76 : i32) : i32
// CHECK-DAG: %[[AC:[^ ]+]] = arith.constant 2183169056 : i64
// CHECK-DAG: %[[CA:[^ ]+]] = arith.constant 2182121504 : i64
// CHECK-DAG: %[[FIRST:[^ ]+]] = arith.constant 32 : i64
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[AC]], %[[FIRST]], %[[F76]])
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[CA]], {{.*}}, %[[F76]])
// CHECK: llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%[[AC]], {{.*}}, %[[F76]])
func.func @reduce_groups() -> i64 attributes {llvm.emit_c_interface} {
  %output = arith.constant 34 : i64
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 2 : i64, group = 0 : i64, first = true}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 34>, destination = 33 : i64}> {rows = 2 : i64, group = 1 : i64, first = false}
  buckyball.custom "add_f32" <{sources = array<i64: 32, 33>, destination = 34 : i64}> {rows = 2 : i64, group = 2 : i64, first = false}
  return %output : i64
}
