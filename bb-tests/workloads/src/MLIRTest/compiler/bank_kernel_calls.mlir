// RUN: buddy-opt %s --target=toy -assign-physical-banks | FileCheck %s
// CHECK-LABEL: func.func @group_output
// CHECK: call @mock
// CHECK: buckyball.mvout {{.*}} <group = 1>
// CHECK-COUNT-2: buckyball.mset {{.*}} <alloc = false
// CHECK-NOT: buckyball.bank_
// CHECK: return
func.func @group_output(%output: memref<1x4xf32>) {
  %read = buckyball.bank_alloc <col = 2>
  %write = buckyball.bank_alloc <col = 3>
  %one = arith.constant 1 : i64
  %r, %w = buckyball.bank_kernel @mock %read %write (%one) : i64
  %stored = buckyball.bank_mvout %output %w %one %one <group = 1> : memref<1x4xf32> i64 i64 i64
  buckyball.bank_release %r : i64
  buckyball.bank_release %stored : i64
  return
}
