// RUN: buddy-opt %s --target=toy -lower-buckyball -canonicalize | FileCheck %s
// CHECK-LABEL: func.func @selected
// CHECK-DAG: %[[MODE:.*]] = arith.constant -8935140561191436288 : i64
// CHECK: arith.ori {{.*}}, %[[MODE]]
// CHECK: "buckyball.intr.mvin"
// CHECK: llvm.call @dma_touch_mvout_group
// CHECK: arith.ori {{.*}}, %[[MODE]]
// CHECK: "buckyball.intr.mvout"
func.func @selected(%input: memref<16x32xi8>, %output: memref<16x32xi8>, %bank: i64) {
  %depth = arith.constant 16 : i64
  %stride = arith.constant 2 : i64
  buckyball.mvin %input %bank %depth %stride <group = 1> : memref<16x32xi8> i64 i64 i64
  buckyball.mvout %output %bank %depth %stride <group = 1> : memref<16x32xi8> i64 i64 i64
  return
}

// CHECK-LABEL: func.func @whole
// CHECK-NOT: dma_touch_mvout_group
// CHECK: "buckyball.intr.mvin"
// CHECK: llvm.call @dma_touch_mvout
// CHECK: "buckyball.intr.mvout"
func.func @whole(%input: memref<16x48xi8>, %output: memref<16x48xi8>, %bank: i64) {
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  buckyball.mvin %input %bank %depth %stride : memref<16x48xi8> i64 i64 i64
  buckyball.mvout %output %bank %depth %stride : memref<16x48xi8> i64 i64 i64
  return
}
