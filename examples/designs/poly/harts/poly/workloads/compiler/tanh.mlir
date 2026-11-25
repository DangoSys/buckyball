// CHECK: hart.launch
// CHECK: func.func private @operator_0(
// CHECK: math.fma
// CHECK-NOT: math.tanh
// CHECK: return
#id = affine_map<(m,n)->(m,n)>
func.func @tanh_rows(%input: memref<?x?xf32, strided<[?,1], offset: ?>>, %out: memref<?x?xf32, strided<[?,1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<?x?xf32, strided<[?,1], offset: ?>> to tensor<?x?xf32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %rows = tensor.dim %a, %zero : tensor<?x?xf32>
  %width = tensor.dim %a, %one : tensor<?x?xf32>
  %empty = tensor.empty(%rows, %width) : tensor<?x?xf32>
  %result = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel"]}
      ins(%a : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
  ^bb0(%x:f32,%unused:f32):
    %value = math.tanh %x : f32
    linalg.yield %value : f32
  } -> tensor<?x?xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<?x?xf32>,memref<?x?xf32, strided<[?,1], offset: ?>>) -> ()
  return
}
