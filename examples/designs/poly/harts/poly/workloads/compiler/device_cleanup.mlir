// CHECK-LABEL: func.func @program(
// CHECK: call @program_start
// CHECK: memref.subview
// CHECK: call @program_dispatch(%c0_i64
// CHECK: call @program_dispatch(%c1_i64
// CHECK: call @program_dispatch(%c2_i64
// CHECK-NOT: call @program_copy_job
// CHECK: call @program_finish
// CHECK: memref.dealloc
// CHECK: return
// Entry 2 on hart 2 reads what entry 0 on hart 1 wrote: it waits for one
// wakeup, from entry 0.
// CHECK: llvm.mlir.global internal constant @program_table(dense<[3, 0, 3, 3, 0, 2, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 4, 0, 3, 1, 0, 1, 0, 1, 0, 1, 0, 2, 8, 1, 3, 1, 0, 1, 0, 1, 0, 1, 0, 2]>
func.func private @compute(memref<1x1x?x128xf32>, memref<1x1x?x128xf32>)
func.func @program(%input: memref<1x8x?x128xf32>) -> memref<1x2x?x128xf32> attributes {device.hart = 0 : i64} {
  %axis = arith.constant 2 : index
  %size = memref.dim %input, %axis : memref<1x8x?x128xf32>
  %slice = memref.subview %input[0, 3, 0, 0] [1, 1, %size, 128] [1, 1, 1, 1] : memref<1x8x?x128xf32> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>
  %buffer = memref.alloc(%size) alignment = 64 : memref<1x1x?x128xf32>
  device.launch @compute(%slice) outs(%buffer) <hart = 1> : (memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>, memref<1x1x?x128xf32>) -> ()
  %result = memref.alloc(%size) alignment = 64 : memref<1x2x?x128xf32>
  %left = memref.subview %result[0, 0, 0, 0] [1, 1, %size, 128] [1, 1, 1, 1] : memref<1x2x?x128xf32> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>>
  %right = memref.subview %result[0, 1, 0, 0] [1, 1, %size, 128] [1, 1, 1, 1] : memref<1x2x?x128xf32> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>
  device.launch @compute(%buffer) outs(%left) <hart = 1> : (memref<1x1x?x128xf32>, memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>>) -> ()
  device.launch @compute(%buffer) outs(%right) <hart = 2> : (memref<1x1x?x128xf32>, memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>) -> ()
  memref.copy %left, %left : memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>>
  return %result : memref<1x2x?x128xf32>
}
