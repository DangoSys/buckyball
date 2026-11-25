module {
  func.func private @copy4(%input: memref<4xi32>, %out: memref<4xi32>) {
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
  func.func @edge_after(%input: memref<4xi32>, %middle: memref<4xi32>, %out: memref<4xi32>) attributes {llvm.emit_c_interface} {
    hart.launch @copy4(%input) outs(%middle) <thread = 0> : (memref<4xi32>,memref<4xi32>) -> ()
    hart.launch @copy4(%middle) outs(%out) <thread = 1> : (memref<4xi32>,memref<4xi32>) -> ()
    return
  }
  func.func @edge_transfer(%input: memref<4xi32>, %out: memref<4xi32>) attributes {llvm.emit_c_interface} {
    %middle = memref.alloc() alignment = 16 : memref<4xi32>
    hart.launch @copy4(%input) outs(%middle) <thread = 0> : (memref<4xi32>,memref<4xi32>) -> ()
    hart.launch @copy4(%middle) outs(%out) <thread = 1> : (memref<4xi32>,memref<4xi32>) -> ()
    memref.dealloc %middle : memref<4xi32>
    return
  }
}
