func.func @quantize(%input: tensor<?x544xf32>, %out: tensor<?x561xi8>) -> tensor<?x561xi8> attributes {llvm.emit_c_interface} {
  %q = "frontend.fp32_to_mxint8"(%input) {axis = 1 : i64} : (tensor<?x544xf32>) -> tensor<?x561xi8>
  %written = bufferization.materialize_in_destination %q in %out : (tensor<?x561xi8>, tensor<?x561xi8>) -> tensor<?x561xi8>
  return %written : tensor<?x561xi8>
}
