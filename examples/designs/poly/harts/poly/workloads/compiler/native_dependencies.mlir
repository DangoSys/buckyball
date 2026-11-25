module {
  func.func private @producer(%input: memref<4xi32>, %out: memref<4xi32>) {
    %one = arith.constant 1 : i64
    %alloc = arith.constant 1057 : i64
    %mvin = arith.constant 1074790400 : i64
    %mvout = arith.constant 1073741825 : i64
    %stride = arith.constant 549755813888 : i64
    %zero = arith.constant 0 : i64
    %f32 = arith.constant 32 : i32
    %f33 = arith.constant 33 : i32
    %f16 = arith.constant 16 : i32
    %src = memref.extract_aligned_pointer_as_index %input : memref<4xi32> -> index
    %dst = memref.extract_aligned_pointer_as_index %out : memref<4xi32> -> index
    %s = arith.index_cast %src : index to i64
    %d = arith.index_cast %dst : index to i64
    %read = arith.ori %s, %stride : i64
    %write = arith.ori %d, %stride : i64
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one, %alloc, %f32) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvin, %read, %f33) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvout, %write, %f16) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one, %zero, %f32) : (i64,i64,i32) -> ()
    return
  }
  func.func private @prepared(%input: memref<4xi32>, %out: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %inc = arith.constant 1 : i32
    %loaded = memref.load %input[%c0] : memref<4xi32>
    %value = arith.addi %loaded, %inc : i32
    %scratch = memref.alloc() alignment = 128 : memref<4xi32>
    memref.store %value, %scratch[%c0] : memref<4xi32>
    memref.store %value, %scratch[%c1] : memref<4xi32>
    memref.store %value, %scratch[%c2] : memref<4xi32>
    memref.store %value, %scratch[%c3] : memref<4xi32>
    %one = arith.constant 1 : i64
    %alloc = arith.constant 1057 : i64
    %mvin = arith.constant 1074790400 : i64
    %mvout = arith.constant 1073741825 : i64
    %stride = arith.constant 549755813888 : i64
    %zero = arith.constant 0 : i64
    %f32 = arith.constant 32 : i32
    %f33 = arith.constant 33 : i32
    %f16 = arith.constant 16 : i32
    %src = memref.extract_aligned_pointer_as_index %scratch : memref<4xi32> -> index
    %dst = memref.extract_aligned_pointer_as_index %out : memref<4xi32> -> index
    %s = arith.index_cast %src : index to i64
    %d = arith.index_cast %dst : index to i64
    %read = arith.ori %s, %stride : i64
    %write = arith.ori %d, %stride : i64
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one, %alloc, %f32) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvin, %read, %f33) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvout, %write, %f16) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one, %zero, %f32) : (i64,i64,i32) -> ()
    memref.dealloc %scratch : memref<4xi32>
    return
  }
  func.func private @completed(%input: memref<4xi32>, %out: memref<4xi32>) {
    %scratch = memref.alloc() alignment = 64 : memref<4xi32>
    %one = arith.constant 1 : i64
    %alloc = arith.constant 1057 : i64
    %mvin = arith.constant 1074790400 : i64
    %mvout = arith.constant 1073741825 : i64
    %stride = arith.constant 549755813888 : i64
    %zero = arith.constant 0 : i64
    %f32 = arith.constant 32 : i32
    %f33 = arith.constant 33 : i32
    %f16 = arith.constant 16 : i32
    %src = memref.extract_aligned_pointer_as_index %input : memref<4xi32> -> index
    %dst = memref.extract_aligned_pointer_as_index %scratch : memref<4xi32> -> index
    %s = arith.index_cast %src : index to i64
    %d = arith.index_cast %dst : index to i64
    %read = arith.ori %s, %stride : i64
    %write = arith.ori %d, %stride : i64
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one, %alloc, %f32) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvin, %read, %f33) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvout, %write, %f16) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one, %zero, %f32) : (i64,i64,i32) -> ()
    memref.copy %scratch, %out {hart.after} : memref<4xi32> to memref<4xi32>
    memref.dealloc %scratch : memref<4xi32>
    return
  }
  func.func @compute_prep(%input: memref<4xi32>, %middle: memref<4xi32>, %out: memref<4xi32>) attributes {llvm.emit_c_interface} {
    hart.launch @producer(%input) outs(%middle) <thread = 0> : (memref<4xi32>,memref<4xi32>) -> ()
    hart.launch @prepared(%middle) outs(%out) <thread = 1> : (memref<4xi32>,memref<4xi32>) -> ()
    return
  }
  func.func @compute_after(%input: memref<4xi32>, %middle: memref<4xi32>, %out: memref<4xi32>) attributes {llvm.emit_c_interface} {
    hart.launch @completed(%input) outs(%middle) <thread = 0> : (memref<4xi32>,memref<4xi32>) -> ()
    hart.launch @producer(%middle) outs(%out) <thread = 1> : (memref<4xi32>,memref<4xi32>) -> ()
    return
  }
}
