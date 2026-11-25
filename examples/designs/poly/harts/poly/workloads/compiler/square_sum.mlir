// CHECK-NOT: math.fpowi
// CHECK-LABEL: func.func private @operator_0(
// CHECK: arith.mulf {{.*}} : vector<32xf32>
// CHECK-NOT: func.func
// CHECK: vector.reduction
// CHECK: return
#id = affine_map<(m,n)->(m,n)>
func.func @square_sum(%input: memref<?x?xf32, strided<[?,1], offset: ?>>, %out: memref<?xf32, strided<[1], offset: ?>>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<?x?xf32, strided<[?,1], offset: ?>> to tensor<?x?xf32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %rows = tensor.dim %a, %zero : tensor<?x?xf32>
  %width = tensor.dim %a, %one : tensor<?x?xf32>
  %two = arith.constant 2 : i32
  %empty = tensor.empty(%rows, %width) : tensor<?x?xf32>
  %squares = linalg.generic {indexing_maps=[#id,#id],iterator_types=["parallel","parallel"]}
      ins(%a : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
  ^bb0(%x:f32,%unused:f32):
    %squared = math.fpowi %x, %two : f32, i32
    linalg.yield %squared : f32
  } -> tensor<?x?xf32>
  %result = tensor.empty(%rows) : tensor<?xf32>
  %z = arith.constant 0.0 : f32
  %init = linalg.fill ins(%z : f32) outs(%result : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.reduce ins(%squares : tensor<?x?xf32>) outs(%init : tensor<?xf32>) dimensions = [1]
      (%x:f32,%acc:f32) {
    %next = arith.addf %acc, %x : f32
    linalg.yield %next : f32
  }
  bufferization.materialize_in_destination %sum in writable %out : (tensor<?xf32>,memref<?xf32, strided<[1], offset: ?>>) -> ()
  return
}
