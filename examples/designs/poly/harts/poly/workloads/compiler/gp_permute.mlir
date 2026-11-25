// CHECK: hart.launch
// CHECK-NOT: linalg.transpose
func.func @permute_rows(%input: memref<2x?x16xf32, strided<[?,16,1], offset: ?>>, %out: memref<?x2x16xf32, strided<[?,16,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<2x?x16xf32, strided<[?,16,1], offset: ?>> to tensor<2x?x16xf32>
  %one = arith.constant 1 : index
  %rows = tensor.dim %a, %one : tensor<2x?x16xf32>
  %empty = tensor.empty(%rows) : tensor<?x2x16xf32>
  %result = linalg.transpose ins(%a : tensor<2x?x16xf32>) outs(%empty : tensor<?x2x16xf32>) permutation = [1,0,2]
  bufferization.materialize_in_destination %result in writable %out : (tensor<?x2x16xf32>, memref<?x2x16xf32, strided<[?,16,1], offset: ?>>) -> ()
  return
}
