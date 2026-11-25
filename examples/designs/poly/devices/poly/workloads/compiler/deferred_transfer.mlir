// CHECK-LABEL: func.func @program(
// CHECK: constant @program_transfer_job
// CHECK: call @program_dispatch(%c0_i64
// CHECK: call @program_dispatch(%c1_i64
// CHECK: call @program_dispatch(%c2_i64
// CHECK-NOT: call @device_copy
// CHECK: call @program_finish
func.func private @compute(memref<4xf32>, memref<4xf32>)
func.func @program(%input: memref<4xf32>, %output: memref<4xf32>) attributes {device.hart = 0 : i64} {
  %source = memref.alloc() : memref<4xf32>
  device.launch @compute(%input) outs(%source) <hart = 0> : (memref<4xf32>, memref<4xf32>) -> ()
  %received = memref.alloc() : memref<4xf32>
  device.transfer %source to(%received) <from = 0, to = 1> {device.defer} : (memref<4xf32>, memref<4xf32>) -> ()
  device.launch @compute(%received) outs(%output) <hart = 1> : (memref<4xf32>, memref<4xf32>) -> ()
  return
}

// A reader of two deferred sends starts after the first sender releases it.
// Waiting for both would prevent it from draining the first sender's ring.
// CHECK-LABEL: func.func @two_sends(
// CHECK: call @program_dispatch(%c0_i64
// CHECK: call @program_dispatch(%c1_i64
// CHECK: call @program_dispatch(%c2_i64
// CHECK: call @program_dispatch(%c3_i64
// CHECK: call @program_finish
func.func private @produce_pair(memref<4xf32>, memref<4xf32>, memref<4xf32>)
func.func private @consume_pair(memref<4xf32>, memref<4xf32>, memref<4xf32>)
func.func @two_sends(%input: memref<4xf32>, %output: memref<4xf32>) attributes {device.hart = 0 : i64} {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  device.launch @produce_pair(%input) outs(%a, %b) <hart = 0> : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
  %ra = memref.alloc() : memref<4xf32>
  %rb = memref.alloc() : memref<4xf32>
  device.transfer %a to(%ra) <from = 0, to = 1> {device.defer} : (memref<4xf32>, memref<4xf32>) -> ()
  device.transfer %b to(%rb) <from = 0, to = 1> {device.defer} : (memref<4xf32>, memref<4xf32>) -> ()
  device.launch @consume_pair(%ra, %rb) outs(%output) <hart = 1> : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
  return
}

// CHECK-LABEL: func.func private @program_transfer_job
// CHECK: return
// CHECK: @program_table(dense<[3, 1, 2,
// CHECK: @two_sends_table(dense<[4, 2, 2, 0, 3,
// CHECK-SAME: 1, 11, 1, 4,
