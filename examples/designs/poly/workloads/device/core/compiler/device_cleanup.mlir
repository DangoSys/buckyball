// CHECK-LABEL: func.func @program(
// CHECK: memref.subview
// CHECK: call @program_job_begin
// CHECK: call @program_job_submit
// CHECK: call @program_job_begin
// CHECK: call @program_job_submit
// CHECK-NOT: call @program_copy
// CHECK: call @program_cleanup_begin
// CHECK: llvm.extractvalue {{.*}}[0]
// CHECK: call @program_job_argument
// CHECK: call @program_job_submit
// CHECK: return
func.func private @compute(memref<1x1x?x128xf32>, memref<1x1x?x128xf32>)
func.func @program(%input: memref<1x8x?x128xf32>) -> memref<1x2x?x128xf32> attributes {device.core = 0 : i64} {
  %axis = arith.constant 2 : index
  %size = memref.dim %input, %axis : memref<1x8x?x128xf32>
  %slice = memref.subview %input[0, 3, 0, 0] [1, 1, %size, 128] [1, 1, 1, 1] : memref<1x8x?x128xf32> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>
  %buffer = memref.alloc(%size) alignment = 64 : memref<1x1x?x128xf32>
  device.launch @compute(%slice) outs(%buffer) <core = 1> : (memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>, memref<1x1x?x128xf32>) -> ()
  %result = memref.alloc(%size) alignment = 64 : memref<1x2x?x128xf32>
  %left = memref.subview %result[0, 0, 0, 0] [1, 1, %size, 128] [1, 1, 1, 1] : memref<1x2x?x128xf32> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>>
  %right = memref.subview %result[0, 1, 0, 0] [1, 1, %size, 128] [1, 1, 1, 1] : memref<1x2x?x128xf32> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>
  device.launch @compute(%buffer) outs(%left) <core = 1> : (memref<1x1x?x128xf32>, memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>>) -> ()
  device.launch @compute(%buffer) outs(%right) <core = 2> : (memref<1x1x?x128xf32>, memref<1x1x?x128xf32, strided<[?, ?, 128, 1], offset: ?>>) -> ()
  memref.copy %left, %left : memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>> to memref<1x1x?x128xf32, strided<[?, ?, 128, 1]>>
  return %result : memref<1x2x?x128xf32>
}
