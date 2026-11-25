// HART-LABEL: func.func @compute_integer_softmax
// HART: hart.launch
// HART-NOT: math.exp
// HART-NOT: linalg.generic
// HART: return
// ACC-NOT: scf.for
// ACC: bbacc.bank_mvin {{.*}}depth = 1024
// ACC: scf.for
// ACC: intsoftmax_ball.bank_softmax
// ACC-NOT: frontend.integer_softmax
// BUFFER-NOT: memref.copy
func.func @compute_integer_softmax(%scores: tensor<?x?xf32>, %table: tensor<4096xi32>, %output: tensor<?x?xf32> {bufferization.writable = true}) -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
  %0 = "frontend.integer_softmax"(%scores, %table) : (tensor<?x?xf32>, tensor<4096xi32>) -> tensor<?x?xf32>
  %1 = bufferization.materialize_in_destination %0 in %output : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
