// CHECK-LABEL: llvm.func @loop_dim
// CHECK-NOT: llvm.alloca
// CHECK: llvm.call @consume
// CHECK-NOT: llvm.alloca
// CHECK: llvm.return
func.func private @consume(index)
func.func @loop_dim(%source: memref<?x?xf32>, %count: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %count step %one {
    %rows = memref.dim %source, %zero : memref<?x?xf32>
    func.call @consume(%rows) : (index) -> ()
  }
  return
}
