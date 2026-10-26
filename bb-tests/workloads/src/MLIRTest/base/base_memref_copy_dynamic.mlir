// The base workload pipeline explicitly runs lower-buckyball before MemRefToLLVM.
// Static strides remain visible to Buckyball's own export patterns.
func.func @copy_unit(%input: memref<?xi8, strided<[1], offset: ?>>,
                     %output: memref<?xi8, strided<[1], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  memref.copy %input, %output : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[1], offset: ?>>
  return
}

func.func @copy_gap(%input: memref<?xi8, strided<[1], offset: ?>>,
                    %output: memref<?xi8, strided<[2], offset: ?>>)
    attributes {llvm.emit_c_interface} {
  memref.copy %input, %output : memref<?xi8, strided<[1], offset: ?>> to memref<?xi8, strided<[2], offset: ?>>
  return
}
