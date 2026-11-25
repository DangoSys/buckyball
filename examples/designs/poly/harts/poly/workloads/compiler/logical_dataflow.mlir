// CHECK-LABEL: func.func @positive(
// CHECK: call @stream_begin
// CHECK: call @produce_launch
// CHECK: call @consume_launch
// CHECK: call @stream_transfer
// CHECK: call @stream_submit
// CHECK: call @stream_wait
// CHECK: return
// CHECK-LABEL: func.func @cpu_reader(
// CHECK: call @stream_begin
// CHECK-NOT: call @stream_transfer
// CHECK: call @stream_wait
// CHECK: memref.load
// CHECK-NOT: call @stream_transfer
// CHECK: return
// CHECK-LABEL: func.func @explicit_cpu_reader(
// CHECK: call @stream_begin
// CHECK-NOT: call @stream_transfer
// CHECK: call @stream_wait
// CHECK: memref.load
// CHECK-NOT: call @stream_transfer
// CHECK: return
func.func private @produce(%input: memref<256xi8>, %output: memref<256xi8>) attributes {bbacc.target = "attention"} {
  %zero = arith.constant 0 : i64
  %allocate = arith.constant 1057 : i64
  %stride = arith.constant 549755813888 : i64
  %read = arith.constant 17179869184 : i64
  %set = arith.constant 32 : i32
  %load = arith.constant 33 : i32
  %store = arith.constant 16 : i32
  %source = memref.extract_aligned_pointer_as_index %input : memref<256xi8> -> index
  %target = memref.extract_aligned_pointer_as_index %output : memref<256xi8> -> index
  %a = arith.index_cast %source : index to i64
  %b = arith.index_cast %target : index to i64
  %src = arith.ori %a, %stride : i64
  %dst = arith.ori %b, %stride : i64
  llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%zero, %allocate, %set) : (i64, i64, i32) -> ()
  llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%read, %src, %load) : (i64, i64, i32) -> ()
  llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%read, %dst, %store) : (i64, i64, i32) -> ()
  llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%zero, %zero, %set) : (i64, i64, i32) -> ()
  return
}
func.func private @consume(%input: memref<1x1x256xi8>, %output: memref<256xi8>) attributes {bbacc.target = "attention"} {
  %flat = memref.collapse_shape %input [[0, 1, 2]] : memref<1x1x256xi8> into memref<256xi8>
  call @produce(%flat, %output) : (memref<256xi8>, memref<256xi8>) -> ()
  return
}
func.func @positive(%input: memref<256xi8>, %output: memref<256xi8>) {
  %temporary = memref.alloc() alignment = 64 : memref<256xi8>
  %logical = memref.alloc() alignment = 64 : memref<256xi8>
  %received = memref.alloc() alignment = 64 : memref<1x1x256xi8>
  hart.launch @produce(%input) outs(%temporary) <thread = 0> : (memref<256xi8>, memref<256xi8>) -> ()
  hart.transfer %temporary to(%logical) <from = 0> : (memref<256xi8>, memref<256xi8>) -> ()
  %view = memref.reinterpret_cast %logical to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<256xi8> to memref<1x1x256xi8>
  hart.transfer %view to(%received) <to = 1> : (memref<1x1x256xi8>, memref<1x1x256xi8>) -> ()
  hart.launch @consume(%received) outs(%output) <thread = 1> : (memref<1x1x256xi8>, memref<256xi8>) -> ()
  memref.dealloc %temporary : memref<256xi8>
  memref.dealloc %logical : memref<256xi8>
  memref.dealloc %received : memref<1x1x256xi8>
  return
}
func.func @cpu_reader(%input: memref<256xi8>, %output: memref<256xi8>) {
  %zero = arith.constant 0 : index
  %temporary = memref.alloc() alignment = 64 : memref<256xi8>
  hart.launch @produce(%input) outs(%temporary) <thread = 0> : (memref<256xi8>, memref<256xi8>) -> ()
  %view = memref.reinterpret_cast %temporary to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<256xi8> to memref<1x1x256xi8>
  hart.launch @consume(%view) outs(%output) <thread = 1> : (memref<1x1x256xi8>, memref<256xi8>) -> ()
  %value = memref.load %temporary[%zero] : memref<256xi8>
  memref.store %value, %output[%zero] : memref<256xi8>
  memref.dealloc %temporary : memref<256xi8>
  return
}
func.func @explicit_cpu_reader(%input: memref<256xi8>, %output: memref<256xi8>) {
  %zero = arith.constant 0 : index
  %temporary = memref.alloc() alignment = 64 : memref<256xi8>
  %received = memref.alloc() alignment = 64 : memref<1x1x256xi8>
  hart.launch @produce(%input) outs(%temporary) <thread = 0> : (memref<256xi8>, memref<256xi8>) -> ()
  %view = memref.reinterpret_cast %temporary to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<256xi8> to memref<1x1x256xi8>
  hart.transfer %view to(%received) <from = 0, to = 1> : (memref<1x1x256xi8>, memref<1x1x256xi8>) -> ()
  hart.launch @consume(%received) outs(%output) <thread = 1> : (memref<1x1x256xi8>, memref<256xi8>) -> ()
  %value = memref.load %temporary[%zero] : memref<256xi8>
  memref.store %value, %output[%zero] : memref<256xi8>
  memref.dealloc %temporary : memref<256xi8>
  memref.dealloc %received : memref<1x1x256xi8>
  return
}
