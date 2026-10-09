// RUN: buddy-opt %s --target=toy -assign-physical-banks | FileCheck %s

// CHECK-LABEL: func.func @release_fragmented
// CHECK: buckyball.mset {{.*}} {{.*}}col = 14
// CHECK: %[[TARGET:.*]] = arith.constant 2 : i64
// CHECK: buckyball.mset_transfer {{.*}} %[[TARGET]]
// CHECK: buckyball.mset {{.*}} <alloc = false
// CHECK: buckyball.mset {{.*}} {{.*}}col = 15
// CHECK-NOT: buckyball.bank_
// CHECK: return
func.func @release_fragmented() {
  %source = buckyball.bank_alloc
  %keep = buckyball.bank_alloc
  %target = buckyball.bank_alloc <col = 14>
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  buckyball.bank_release %aggregate : i64
  %all_free = buckyball.bank_alloc <col = 15>
  buckyball.bank_release %all_free : i64
  buckyball.bank_release %keep : i64
  return
}

// Virtual target 2 initially owns physical group 0. Later allocations own
// physical groups 1 and 2, with virtual IDs 0 and 1, preserving target 2.
// CHECK-LABEL: func.func @virtual_physical_collision
// CHECK: buckyball.mset
// CHECK: %[[START:.*]] = arith.constant 0 : i64
// CHECK: %[[TARGET2:.*]] = arith.constant 2 : i64
// CHECK: buckyball.mset_transfer %[[START]] %[[TARGET2]]
// CHECK: %[[FIRST:.*]] = arith.constant 0 : i64
// CHECK: buckyball.mset %[[FIRST]]
// CHECK: %[[SECOND:.*]] = arith.constant 1 : i64
// CHECK: buckyball.mset %[[SECOND]]
// CHECK: buckyball.mset {{.*}} <alloc = false
// CHECK: %[[THIRD:.*]] = arith.constant 2 : i64
// CHECK: buckyball.mset %[[THIRD]]
// CHECK-NOT: buckyball.bank_
// CHECK: return
func.func @virtual_physical_collision() {
  %source = buckyball.bank_alloc
  %target = arith.constant 2 : i64
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  %first = buckyball.bank_alloc
  %second = buckyball.bank_alloc
  buckyball.bank_release %aggregate : i64
  %third = buckyball.bank_alloc
  buckyball.bank_release %first : i64
  buckyball.bank_release %second : i64
  buckyball.bank_release %third : i64
  return
}

// CHECK-LABEL: func.func @private_limit
// CHECK: buckyball.mset
// CHECK: %[[MAX:.*]] = arith.constant 15 : i64
// CHECK: buckyball.mset_transfer {{.*}} %[[MAX]]
// CHECK: buckyball.mset {{.*}} <alloc = false
// CHECK: return
func.func @private_limit() {
  %source = buckyball.bank_alloc
  %target = arith.constant 15 : i64
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  buckyball.bank_release %aggregate : i64
  return
}
