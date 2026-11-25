// CHECK: bbacc.bank_run_operator
#input = affine_map<(d0, d1) -> (d0, d1)>
#output = affine_map<(d0, d1) -> (d0)>
func.func @compute(%input: memref<?x?xf32>, %out: memref<?xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<?x?xf32> to tensor<?x?xf32>
  %zero = arith.constant 0 : index
  %rows = tensor.dim %a, %zero : tensor<?x?xf32>
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %value = arith.constant 0.0 : f32
  %init = linalg.fill ins(%value : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {indexing_maps = [#input, #output], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<?x?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %sum: f32):
    %next = arith.addf %sum, %x : f32
    linalg.yield %next : f32
  } -> tensor<?xf32>
  bufferization.materialize_in_destination %sum in writable %out : (tensor<?xf32>, memref<?xf32>) -> ()
  return
}

func.func @maximum(%input: memref<?x?xf32>, %out: memref<?xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<?x?xf32> to tensor<?x?xf32>
  %zero = arith.constant 0 : index
  %rows = tensor.dim %a, %zero : tensor<?x?xf32>
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %value = arith.constant 0xFF800000 : f32
  %init = linalg.fill ins(%value : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {indexing_maps = [#input, #output], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<?x?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %sum: f32):
    %next = arith.maximumf %sum, %x : f32
    linalg.yield %next : f32
  } -> tensor<?xf32>
  bufferization.materialize_in_destination %sum in writable %out : (tensor<?xf32>, memref<?xf32>) -> ()
  return
}
