// RUN: buddy-opt %s --target=main -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @reduce_groups
// CHECK-DAG: %[[AC:[^ ]+]] = arith.constant 2183169056 : i64
// CHECK-DAG: %[[CA:[^ ]+]] = arith.constant 2182121504 : i64
// CHECK-DAG: %[[FIRST:[^ ]+]] = arith.constant 32 : i64
// CHECK: buckyball.intr.custom %[[AC]], %[[FIRST]] <funct7 = 76>
// CHECK: buckyball.intr.custom %[[CA]], {{.*}} <funct7 = 76>
// CHECK: buckyball.intr.custom %[[AC]], {{.*}} <funct7 = 76>
func.func @reduce_groups() -> i64 attributes {llvm.emit_c_interface} {
  %source = arith.constant 32 : i64
  %accumulator = arith.constant 33 : i64
  %output = arith.constant 34 : i64
  buckyball.f32add %source, %accumulator, %output <rows = 2, group = 0, first = true> : i64
  buckyball.f32add %source, %output, %accumulator <rows = 2, group = 1, first = false> : i64
  buckyball.f32add %source, %accumulator, %output <rows = 2, group = 2, first = false> : i64
  return %output : i64
}
