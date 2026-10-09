// RUN: buddy-opt %s --target=toy -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @dynamic
// CHECK-SAME: %[[SRC:.*]]: i64, %[[DST:.*]]: i64
// CHECK: llvm.call @dma_bank_transfer(%[[SRC]], %[[DST]]) : (i64, i64) -> ()
// CHECK: arith.shli %[[DST]]
// CHECK: arith.ori %[[SRC]]
// CHECK: "buckyball.intr.mset"
func.func @dynamic(%source: i64, %target: i64) {
  buckyball.mset_transfer %source %target : i64 i64
  return
}

// CHECK-LABEL: func.func @constant
// CHECK-DAG: %[[MODE:.*]] = arith.constant 4096 : i64
// CHECK-DAG: %[[PACKED:.*]] = arith.constant 17825793 : i64
// CHECK: llvm.call @dma_bank_transfer
// CHECK: "buckyball.intr.mset"(%[[PACKED]], %[[MODE]])
func.func @constant() {
  %source = arith.constant 1 : i64
  %target = arith.constant 17 : i64
  buckyball.mset_transfer %source %target : i64 i64
  return
}
