func.func private @quantize_rows(%input: tensor<?x1024xf32>) -> tensor<?x1056xi8> {
  %q = "frontend.fp32_to_mxint8"(%input) {axis = 1 : i64} : (tensor<?x1024xf32>) -> tensor<?x1056xi8>
  return %q : tensor<?x1056xi8>
}
func.func @quantize(%input: tensor<?x1024xf32>, %out: tensor<?x1056xi8>) -> tensor<?x1056xi8> attributes {llvm.emit_c_interface} {
  %q = call @quantize_rows(%input) : (tensor<?x1024xf32>) -> tensor<?x1056xi8>
  %written = bufferization.materialize_in_destination %q in %out : (tensor<?x1056xi8>, tensor<?x1056xi8>) -> tensor<?x1056xi8>
  return %written : tensor<?x1056xi8>
}
