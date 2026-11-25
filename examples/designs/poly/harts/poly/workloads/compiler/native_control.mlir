module {
  func.func private @control() {
    %one = arith.constant 1 : i64
    %allocated = arith.constant 1057 : i64
    %zero = arith.constant 0 : i64
    %funct = arith.constant 32 : i32
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one,%allocated,%funct) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one,%zero,%funct) : (i64,i64,i32) -> ()
    return
  }
  func.func private @empty(%out: memref<0xi8>) {
    %scratch = memref.alloc() alignment = 64 : memref<4xi32>
    %index = arith.constant 0 : index
    %value = arith.constant 9 : i32
    memref.store %value, %scratch[%index] : memref<4xi32>
    %last = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %i = %index to %last step %step {
      memref.store %value, %scratch[%i] : memref<4xi32>
    }
    %one = arith.constant 1 : i64
    %allocated = arith.constant 1057 : i64
    %zero = arith.constant 0 : i64
    %funct = arith.constant 32 : i32
    %mvin = arith.constant 1074790400 : i64
    %readFunct = arith.constant 33 : i32
    %stride = arith.constant 549755813888 : i64
    %pointer = memref.extract_aligned_pointer_as_index %scratch : memref<4xi32> -> index
    %address = arith.index_cast %pointer : index to i64
    %read = arith.ori %address, %stride : i64
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one,%allocated,%funct) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%mvin,%read,%readFunct) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%one,%zero,%funct) : (i64,i64,i32) -> ()
    memref.dealloc %scratch : memref<4xi32>
    return
  }
  func.func private @control_input(%input: memref<4xi32>) {
    %bank = arith.constant 2 : i64
    %allocated = arith.constant 1057 : i64
    %zero = arith.constant 0 : i64
    %funct = arith.constant 32 : i32
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%bank,%allocated,%funct) : (i64,i64,i32) -> ()
    llvm.call_intrinsic "llvm.riscv.buckyball.custom"(%bank,%zero,%funct) : (i64,i64,i32) -> ()
    return
  }
  func.func @compute(%input: memref<4xi32>, %out: memref<0xi8>) attributes {llvm.emit_c_interface} {
    hart.launch @control() outs() <thread = 0> : () -> ()
    hart.launch @control_input(%input) outs() <thread = 0> : (memref<4xi32>) -> ()
    hart.launch @empty() outs(%out) <thread = 1> : (memref<0xi8>) -> ()
    return
  }
}
